// ==== Pipeline Scheduler ====
//
// Takes AlgoSteps and generates concrete OpEntry instructions with:
// - Pipeline chunking (C chunks, each on its own stream)
// - Buffer allocation (input/output/scratch for each pipeline chunk)
// - Double-buffered receive for latency hiding

use crate::algo::{AlgoStep, Phase};
use crate::plan::{BufEntry, BufPool, OpEntry, Opcode, ReduceOp};

// ==== Types ====

pub struct ScheduleResult {
    pub ops:         Vec<OpEntry>,
    pub buffers:     Vec<BufEntry>,
    pub num_streams: u8,
}

// ==== Scheduler ====

/// Pipeline-schedule AlgoSteps into concrete OpEntry instructions.
///
/// Buffer layout per pipeline chunk (N = num_ranks):
///   N input  sub-buffers (ring_piece each, one per ring chunk)
///   N output sub-buffers (ring_piece each, one per ring chunk)
///   2 scratch buffers    (ring_piece each, double-buffered receive)
///
/// For each pipeline chunk c (stream c), for each AlgoStep:
///   RS step -> Put(send_chunk) + Signal(dst) + Wait(src) + LocalReduce(recv -> accum)
///   AG step -> Put(send_chunk) + Signal(dst) + Wait(src) + LocalCopy(recv -> output)
pub fn schedule(steps: &[AlgoStep], msg_size: usize, pipeline_chunks: usize) -> ScheduleResult {
    let num_ranks = if steps.is_empty() {
        8
    } else {
        steps.iter().map(|s| s.send_chunk.max(s.recv_chunk) as usize).max().unwrap() + 1
    };

    let chunk_size = msg_size / pipeline_chunks;
    let ring_piece = chunk_size / num_ranks;

    let mut buffers = Vec::new();
    let mut ops = Vec::new();

    for c in 0..pipeline_chunks {
        let stream = c as u8;
        let pipe_offset = c * chunk_size;

        // -- Per-ring-chunk input sub-buffers --
        let input_base = buffers.len() as u16;
        for j in 0..num_ranks {
            buffers.push(BufEntry {
                pool:   BufPool::Input as u32,
                offset: (pipe_offset + j * ring_piece) as u32,
                size:   ring_piece as u32,
            });
        }

        // -- Per-ring-chunk output sub-buffers --
        let output_base = buffers.len() as u16;
        for j in 0..num_ranks {
            buffers.push(BufEntry {
                pool:   BufPool::Output as u32,
                offset: (pipe_offset + j * ring_piece) as u32,
                size:   ring_piece as u32,
            });
        }

        // -- Scratch: double-buffered receive --
        let scratch_a = buffers.len() as u16;
        buffers.push(BufEntry {
            pool:   BufPool::Scratch as u32,
            offset: (c * 2 * ring_piece) as u32,
            size:   ring_piece as u32,
        });
        let scratch_b = buffers.len() as u16;
        buffers.push(BufEntry {
            pool:   BufPool::Scratch as u32,
            offset: ((c * 2 + 1) * ring_piece) as u32,
            size:   ring_piece as u32,
        });

        // -- Generate ops for each algorithm step --
        for (i, step) in steps.iter().enumerate() {
            let recv_buf = if i % 2 == 0 { scratch_a } else { scratch_b };

            match step.phase {
                Phase::ReduceScatter => {
                    // Put: send ring chunk[send_chunk] to dst_rank's scratch
                    ops.push(OpEntry::new(
                        Opcode::Put, stream,
                        input_base + step.send_chunk,
                        recv_buf,
                        step.dst_rank,
                        ReduceOp::Sum, 0, 0,
                    ));
                    ops.push(OpEntry::new(
                        Opcode::Signal, stream,
                        0, 0, step.dst_rank,
                        ReduceOp::Sum, 0, 0,
                    ));
                    ops.push(OpEntry::new(
                        Opcode::Wait, stream,
                        0, 0, step.src_rank,
                        ReduceOp::Sum, 0, 0,
                    ));
                    // LocalReduce: scratch -> input[recv_chunk] (accumulate)
                    ops.push(OpEntry::new(
                        Opcode::LocalReduce, stream,
                        recv_buf,
                        input_base + step.recv_chunk,
                        0,
                        ReduceOp::Sum, 0, 0,
                    ));
                }
                Phase::AllGather => {
                    // Put: send reduced chunk[send_chunk] to dst_rank's scratch
                    ops.push(OpEntry::new(
                        Opcode::Put, stream,
                        input_base + step.send_chunk,
                        recv_buf,
                        step.dst_rank,
                        ReduceOp::Sum, 0, 0,
                    ));
                    ops.push(OpEntry::new(
                        Opcode::Signal, stream,
                        0, 0, step.dst_rank,
                        ReduceOp::Sum, 0, 0,
                    ));
                    ops.push(OpEntry::new(
                        Opcode::Wait, stream,
                        0, 0, step.src_rank,
                        ReduceOp::Sum, 0, 0,
                    ));
                    // LocalCopy: scratch -> output[recv_chunk]
                    ops.push(OpEntry::new(
                        Opcode::LocalCopy, stream,
                        recv_buf,
                        output_base + step.recv_chunk,
                        0,
                        ReduceOp::Sum, 0, 0,
                    ));
                }
            }
        }
    }

    ScheduleResult {
        ops,
        buffers,
        num_streams: pipeline_chunks as u8,
    }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo;

    #[test]
    fn schedule_produces_ops() {
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, 256 * 1024 * 1024, 4);
        assert!(!sched.ops.is_empty());
        assert!(!sched.buffers.is_empty());
    }

    #[test]
    fn schedule_uses_c_streams() {
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, 256 * 1024 * 1024, 4);
        let max_stream = sched.ops.iter().map(|op| op.stream_id).max().unwrap();
        assert_eq!(max_stream, 3); // 4 streams: 0,1,2,3
    }

    #[test]
    fn schedule_buffer_sizes() {
        let msg_size: usize = 256 << 20; // 256MB
        let chunks = 4;
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, msg_size, chunks);

        let chunk_size = msg_size / chunks;
        let ring_piece = chunk_size / 8;

        let scratch_bufs: Vec<_> = sched.buffers.iter()
            .filter(|b| b.pool == BufPool::Scratch as u32)
            .collect();
        // 2 scratch buffers per pipeline chunk = 4*2 = 8
        assert!(scratch_bufs.len() >= chunks * 2);
        for b in &scratch_bufs {
            assert_eq!(b.size as usize, ring_piece);
        }
    }

    #[test]
    fn schedule_op_sequence_per_stream() {
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, 8192, 1); // 1 pipeline chunk

        // 14 ring steps * 4 ops/step = 56 ops on stream 0
        let stream0_ops: Vec<_> = sched.ops.iter()
            .filter(|op| op.stream_id == 0)
            .collect();
        assert_eq!(stream0_ops.len(), 56);
    }
}
