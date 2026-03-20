// ==== Plan IR Types + Compilation Pipeline ====
//
// Serialization format: header ++ buf_entries ++ op_entries (raw repr(C) bytes)
// Both Rust and C++ can read via reinterpret_cast — no IDL needed.

pub const PLAN_MAGIC: u32 = 0x4B4E_4C50; // "PLNK" little-endian
pub const PLAN_VERSION: u16 = 1;

// ==== Enums ====

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Put = 0,
    Signal,
    Wait,
    LocalCopy,
    LocalReduce,
    PutWithSignal,   // fused: Put + Signal
    WaitReduceCopy,  // fused: Wait + Reduce + Copy
    WaitReducePut,   // fused: Wait + Reduce + Put
    Noop,            // sync point
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum = 0,
    Max,
    Min,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufPool {
    Scratch = 0,
    Input,
    Output,
}

// ==== Serialized (repr(C)) Structures ====

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlanHeader {
    pub magic:       u32,      // PLAN_MAGIC
    pub version:     u16,      // PLAN_VERSION
    pub num_ops:     u16,
    pub num_buffers: u16,
    pub num_streams: u8,
    pub num_events:  u8,
    pub num_ranks:   u16,
    pub my_rank:     u16,
    pub flags:       u32,
    pub _reserved:   [u8; 12],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BufEntry {
    pub pool:   u32, // BufPool as u32
    pub offset: u32,
    pub size:   u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OpEntry {
    pub opcode:       u8,
    pub stream_id:    u8,
    pub reduce_op:    u8,
    pub flags:        u8,
    pub src_buf:      u16, // index into buffers[]
    pub dst_buf:      u16, // index into buffers[]
    pub dst_rank:     u16,
    pub wait_event:   u16, // 0 = no wait
    pub signal_event: u16,
    pub _pad:         u16,
}

// ==== ExecutionPlan ====

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub header:  PlanHeader,
    pub buffers: Vec<BufEntry>,
    pub ops:     Vec<OpEntry>,
}

// ==== Constructors ====

impl PlanHeader {
    pub fn new(
        num_ranks: u16,
        my_rank: u16,
        num_streams: u8,
        num_buffers: u16,
        num_ops: u16,
    ) -> Self {
        Self {
            magic: PLAN_MAGIC,
            version: PLAN_VERSION,
            num_ops,
            num_buffers,
            num_streams,
            num_events: 0,
            num_ranks,
            my_rank,
            flags: 0,
            _reserved: [0; 12],
        }
    }
}

impl OpEntry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        opcode: Opcode,
        stream_id: u8,
        src_buf: u16,
        dst_buf: u16,
        dst_rank: u16,
        reduce_op: ReduceOp,
        wait_event: u16,
        signal_event: u16,
    ) -> Self {
        Self {
            opcode:       opcode as u8,
            stream_id,
            reduce_op:    reduce_op as u8,
            flags:        0,
            src_buf,
            dst_buf,
            dst_rank,
            wait_event,
            signal_event,
            _pad:         0,
        }
    }
}

// ==== Serialization ====

/// Safe helper: cast a repr(C) struct to a byte slice.
fn as_bytes<T: Copy>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Safe helper: cast a slice of repr(C) structs to a byte slice.
fn slice_as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}

impl ExecutionPlan {
    /// Serialize to raw bytes: header ++ buffers ++ ops.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            std::mem::size_of::<PlanHeader>()
                + self.buffers.len() * std::mem::size_of::<BufEntry>()
                + self.ops.len() * std::mem::size_of::<OpEntry>(),
        );
        out.extend_from_slice(as_bytes(&self.header));
        out.extend_from_slice(slice_as_bytes(&self.buffers));
        out.extend_from_slice(slice_as_bytes(&self.ops));
        out
    }

    /// Deserialize from raw bytes. Returns None on invalid data.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        let h_size = std::mem::size_of::<PlanHeader>();
        if data.len() < h_size {
            return None;
        }

        // Read header
        let header: PlanHeader =
            unsafe { std::ptr::read_unaligned(data.as_ptr() as *const PlanHeader) };
        if header.magic != PLAN_MAGIC || header.version != PLAN_VERSION {
            return None;
        }

        let b_size = std::mem::size_of::<BufEntry>();
        let o_size = std::mem::size_of::<OpEntry>();
        let expected = h_size + header.num_buffers as usize * b_size + header.num_ops as usize * o_size;
        if data.len() < expected {
            return None;
        }

        // Read buffers
        let buf_start = h_size;
        let buffers: Vec<BufEntry> = (0..header.num_buffers as usize)
            .map(|i| unsafe {
                std::ptr::read_unaligned(data[buf_start + i * b_size..].as_ptr() as *const BufEntry)
            })
            .collect();

        // Read ops
        let op_start = buf_start + header.num_buffers as usize * b_size;
        let ops: Vec<OpEntry> = (0..header.num_ops as usize)
            .map(|i| unsafe {
                std::ptr::read_unaligned(data[op_start + i * o_size..].as_ptr() as *const OpEntry)
            })
            .collect();

        Some(Self { header, buffers, ops })
    }
}

// ==== Compilation Types ====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collective {
    AllReduce,
}

pub struct CompileRequest {
    pub collective:      Collective,
    pub msg_size:        usize,
    pub reduce_op:       ReduceOp,
    pub num_ranks:       usize,
    pub my_rank:         usize,
    pub pipeline_chunks: usize,
}

// ==== Fusion Pass ====

/// Fuse adjacent instructions within each stream.
///
/// Patterns (greedy, longest match first):
///   Wait + LocalReduce + Put   -> WaitReducePut
///   Wait + LocalReduce + LocalCopy -> WaitReduceCopy
///   Put  + Signal              -> PutWithSignal
pub fn fuse(ops: Vec<OpEntry>) -> Vec<OpEntry> {
    let mut result = Vec::with_capacity(ops.len());
    let mut i = 0;

    while i < ops.len() {
        // Try 3-op patterns first
        if i + 2 < ops.len()
            && ops[i].stream_id == ops[i + 1].stream_id
            && ops[i + 1].stream_id == ops[i + 2].stream_id
        {
            let (a, b, c) = (ops[i].opcode, ops[i + 1].opcode, ops[i + 2].opcode);

            // Wait + LocalReduce + Put -> WaitReducePut
            // src=reduce_src(scratch), dst=reduce_dst=put_src(input[chunk])
            // _pad=put_dst on remote(scratch), dst_rank=put destination
            // wait_event=Wait source rank (used by DES engine for signal matching)
            if a == Opcode::Wait as u8
                && b == Opcode::LocalReduce as u8
                && c == Opcode::Put as u8
            {
                result.push(OpEntry {
                    opcode:       Opcode::WaitReducePut as u8,
                    stream_id:    ops[i].stream_id,
                    reduce_op:    ops[i + 1].reduce_op,
                    flags:        0,
                    src_buf:      ops[i + 1].src_buf,
                    dst_buf:      ops[i + 1].dst_buf,
                    dst_rank:     ops[i + 2].dst_rank,
                    wait_event:   ops[i].dst_rank, // Wait source rank
                    signal_event: ops[i + 2].signal_event,
                    _pad:         ops[i + 2].dst_buf, // put dst buf on remote
                });
                i += 3;
                continue;
            }

            // Wait + LocalReduce + LocalCopy -> WaitReduceCopy
            // src=reduce_src(scratch), dst=copy_dst(output[chunk])
            // _pad=reduce_dst(input[chunk]), dst_rank=wait source
            if a == Opcode::Wait as u8
                && b == Opcode::LocalReduce as u8
                && c == Opcode::LocalCopy as u8
            {
                result.push(OpEntry {
                    opcode:       Opcode::WaitReduceCopy as u8,
                    stream_id:    ops[i].stream_id,
                    reduce_op:    ops[i + 1].reduce_op,
                    flags:        0,
                    src_buf:      ops[i + 1].src_buf,
                    dst_buf:      ops[i + 2].dst_buf,
                    dst_rank:     ops[i].dst_rank,
                    wait_event:   ops[i].wait_event,
                    signal_event: ops[i + 2].signal_event,
                    _pad:         ops[i + 1].dst_buf, // reduce dst (input[chunk])
                });
                i += 3;
                continue;
            }
        }

        // Try 2-op pattern: Put + Signal -> PutWithSignal
        if i + 1 < ops.len()
            && ops[i].stream_id == ops[i + 1].stream_id
            && ops[i].opcode == Opcode::Put as u8
            && ops[i + 1].opcode == Opcode::Signal as u8
        {
            result.push(OpEntry {
                opcode:       Opcode::PutWithSignal as u8,
                stream_id:    ops[i].stream_id,
                reduce_op:    ops[i].reduce_op,
                flags:        0,
                src_buf:      ops[i].src_buf,
                dst_buf:      ops[i].dst_buf,
                dst_rank:     ops[i].dst_rank,
                wait_event:   ops[i].wait_event,
                signal_event: ops[i + 1].signal_event,
                _pad:         0,
            });
            i += 2;
            continue;
        }

        // No fusion: emit as-is
        result.push(ops[i]);
        i += 1;
    }

    result
}

// ==== Compile Pipeline ====

/// Top-level compilation: topo -> cost -> algo -> sched -> fuse -> ExecutionPlan
pub fn compile(req: &CompileRequest, topo: &crate::topo::Topology) -> ExecutionPlan {
    use crate::{algo, cost, sched};

    // Step 0: build cost model from topology (used for algo selection in v0.2+)
    let _cost = cost::CostModel::from_topology(topo);

    // Step 1: generate algorithm steps (v0.1: always Ring)
    let steps = algo::ring_allreduce(topo.num_ranks as u16, req.my_rank as u16);

    // Step 2: schedule into concrete ops
    let sched_result = sched::schedule(&steps, req.msg_size, req.pipeline_chunks);

    // Step 3: fuse adjacent instructions
    let fused_ops = fuse(sched_result.ops);

    // Step 4: assemble final plan
    ExecutionPlan {
        header: PlanHeader::new(
            req.num_ranks as u16,
            req.my_rank as u16,
            sched_result.num_streams,
            sched_result.buffers.len() as u16,
            fused_ops.len() as u16,
        ),
        buffers: sched_result.buffers,
        ops: fused_ops,
    }
}

// ==== Opcode Conversion ====

impl Opcode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Put),
            1 => Some(Self::Signal),
            2 => Some(Self::Wait),
            3 => Some(Self::LocalCopy),
            4 => Some(Self::LocalReduce),
            5 => Some(Self::PutWithSignal),
            6 => Some(Self::WaitReduceCopy),
            7 => Some(Self::WaitReducePut),
            8 => Some(Self::Noop),
            _ => None,
        }
    }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn header_is_32_bytes() {
        assert_eq!(mem::size_of::<PlanHeader>(), 32);
    }

    #[test]
    fn buf_entry_is_12_bytes() {
        assert_eq!(mem::size_of::<BufEntry>(), 12);
    }

    #[test]
    fn op_entry_is_16_bytes() {
        assert_eq!(mem::size_of::<OpEntry>(), 16);
    }

    #[test]
    fn header_magic() {
        let h = PlanHeader::new(8, 0, 1, 0, 0);
        assert_eq!(h.magic, PLAN_MAGIC);
        assert_eq!(h.version, PLAN_VERSION);
    }

    #[test]
    fn opcode_values() {
        assert_eq!(Opcode::Put as u8, 0);
        assert_eq!(Opcode::WaitReducePut as u8, 7);
        assert_eq!(Opcode::Noop as u8, 8);
    }

    #[test]
    fn fusion_put_signal() {
        let ops = vec![
            OpEntry::new(Opcode::Put, 0, 0, 1, 1, ReduceOp::Sum, 0, 0),
            OpEntry::new(Opcode::Signal, 0, 0, 0, 1, ReduceOp::Sum, 0, 0),
        ];
        let fused = fuse(ops);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].opcode, Opcode::PutWithSignal as u8);
    }

    #[test]
    fn fusion_wait_reduce_put() {
        let ops = vec![
            OpEntry::new(Opcode::Wait, 0, 0, 0, 7, ReduceOp::Sum, 0, 0),
            OpEntry::new(Opcode::LocalReduce, 0, 0, 1, 0, ReduceOp::Sum, 0, 0),
            OpEntry::new(Opcode::Put, 0, 1, 2, 1, ReduceOp::Sum, 0, 0),
        ];
        let fused = fuse(ops);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].opcode, Opcode::WaitReducePut as u8);
    }

    #[test]
    fn fusion_preserves_unfusable() {
        let ops = vec![
            OpEntry::new(Opcode::Noop, 0, 0, 0, 0, ReduceOp::Sum, 0, 0),
            OpEntry::new(Opcode::Put, 0, 0, 1, 1, ReduceOp::Sum, 0, 0),
        ];
        let fused = fuse(ops);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn compile_produces_valid_plan() {
        use crate::topo::Topology;

        let topo = Topology::hccs_8card();
        let req = CompileRequest {
            collective:      Collective::AllReduce,
            msg_size:        256 << 20,
            reduce_op:       ReduceOp::Sum,
            num_ranks:       8,
            my_rank:         0,
            pipeline_chunks: 4,
        };
        let plan = compile(&req, &topo);
        assert_eq!(plan.header.magic, PLAN_MAGIC);
        assert_eq!(plan.header.num_ranks, 8);
        assert_eq!(plan.header.my_rank, 0);
        assert!(plan.ops.len() > 0);
        assert!(plan.buffers.len() > 0);
        // After fusion, op count should be < 56 * 4 (4 chunks * 56 unfused ops)
        assert!(plan.ops.len() < 56 * 4);
    }

    #[test]
    fn serialize_roundtrip() {
        let plan = ExecutionPlan {
            header: PlanHeader::new(8, 3, 2, 2, 2),
            buffers: vec![
                BufEntry { pool: BufPool::Input as u32, offset: 0, size: 1024 },
                BufEntry { pool: BufPool::Scratch as u32, offset: 0, size: 512 },
            ],
            ops: vec![
                OpEntry::new(Opcode::Put, 0, 0, 1, 1, ReduceOp::Sum, 0, 1),
                OpEntry::new(Opcode::Wait, 0, 0, 0, 7, ReduceOp::Sum, 1, 0),
            ],
        };
        let bytes = plan.serialize();
        let restored = ExecutionPlan::deserialize(&bytes).unwrap();
        assert_eq!(restored.header.num_ranks, 8);
        assert_eq!(restored.header.my_rank, 3);
        assert_eq!(restored.buffers.len(), 2);
        assert_eq!(restored.ops.len(), 2);
        assert_eq!(restored.ops[0].opcode, Opcode::Put as u8);
        assert_eq!(restored.ops[1].opcode, Opcode::Wait as u8);
    }

    /// E2E correctness gate: simulate Ring AllReduce across 8 ranks.
    /// Input: rank r has [(r+1) as f32; 64], expect all ranks get [36.0; 64].
    #[test]
    fn simulate_ring_allreduce() {
        use crate::algo::ring_allreduce;
        use crate::topo::Topology;

        let n = 8usize;
        let chunk_len = 8usize; // 8 f32s per sub-chunk, 64 total

        // Verify compilation succeeds for all 8 ranks
        let topo = Topology::hccs_8card();
        for r in 0..n {
            let req = CompileRequest {
                collective:      Collective::AllReduce,
                msg_size:        n * chunk_len * 4, // 256 bytes
                reduce_op:       ReduceOp::Sum,
                num_ranks:       n,
                my_rank:         r,
                pipeline_chunks: 1,
            };
            let plan = compile(&req, &topo);
            assert_eq!(plan.header.magic, PLAN_MAGIC);
            assert!(!plan.ops.is_empty());
        }

        // Algorithm-level simulation: rank r, chunk k = [(r+1); chunk_len]
        let mut data: Vec<Vec<Vec<f32>>> = (0..n)
            .map(|r| vec![vec![(r + 1) as f32; chunk_len]; n])
            .collect();

        let all_steps: Vec<Vec<_>> = (0..n)
            .map(|r| ring_allreduce(n as u16, r as u16))
            .collect();

        // Execute in lockstep: RS (n-1 steps) + AG (n-1 steps)
        for step_idx in 0..2 * (n - 1) {
            // Snapshot sends before processing receives
            let sends: Vec<Vec<f32>> = (0..n)
                .map(|r| data[r][all_steps[r][step_idx].send_chunk as usize].clone())
                .collect();

            for r in 0..n {
                let step = &all_steps[r][step_idx];
                let from = step.src_rank as usize;
                let ri = step.recv_chunk as usize;

                if step.needs_reduce {
                    for i in 0..chunk_len { data[r][ri][i] += sends[from][i]; }
                } else {
                    data[r][ri] = sends[from].clone();
                }
            }
        }

        // Verify: all ranks, all chunks = [36.0; chunk_len]  (1+2+...+8 = 36)
        for r in 0..n {
            for k in 0..n {
                for i in 0..chunk_len {
                    assert!((data[r][k][i] - 36.0).abs() < 1e-6,
                        "rank {r} chunk {k} elem {i}: got {}", data[r][k][i]);
                }
            }
        }
    }

    /// Plan-level E2E: compile plans for 8 ranks, execute OpEntry instructions
    /// via lockstep simulator, verify AllReduce result.
    /// This validates the FULL compiler pipeline: algo -> sched -> fuse -> execute.
    #[test]
    fn simulate_plan_execution() {
        use crate::topo::Topology;

        let n = 8usize;
        let chunk_f32 = 8usize;                  // f32s per ring chunk
        let msg_size = n * chunk_f32 * 4;         // 256 bytes total
        let nf = msg_size / 4;                    // 64 f32s total

        let topo = Topology::hccs_8card();
        let plans: Vec<_> = (0..n).map(|r| compile(&CompileRequest {
            collective:      Collective::AllReduce,
            msg_size,
            reduce_op:       ReduceOp::Sum,
            num_ranks:       n,
            my_rank:         r,
            pipeline_chunks: 1,
        }, &topo)).collect();

        // All ranks must have the same op count (symmetric schedule)
        for p in &plans { assert_eq!(p.ops.len(), plans[0].ops.len()); }

        // In-place data: Input == Output. Rank r starts with [r+1; nf].
        let mut data: Vec<Vec<f32>> = (0..n)
            .map(|r| vec![(r + 1) as f32; nf])
            .collect();

        // Scratch: size from buffer table
        let scratch_f32 = plans[0].buffers.iter()
            .filter(|b| b.pool == BufPool::Scratch as u32)
            .map(|b| ((b.offset + b.size) as usize + 3) / 4)
            .max().unwrap_or(0);
        let mut scratch: Vec<Vec<f32>> = (0..n)
            .map(|_| vec![0.0f32; scratch_f32])
            .collect();

        // Read from a rank's buffer (Input/Output -> data, Scratch -> scratch)
        fn read_buf(data: &[f32], scratch: &[f32], buf: &BufEntry, sz: usize) -> Vec<f32> {
            let off = buf.offset as usize / 4;
            if buf.pool == BufPool::Scratch as u32 {
                scratch[off..off + sz].to_vec()
            } else {
                data[off..off + sz].to_vec()
            }
        }

        let num_ops = plans[0].ops.len();

        for op_idx in 0..num_ops {
            // Phase 1: local ops + stage puts
            let mut puts: Vec<(usize, usize, Vec<f32>)> = Vec::new(); // (dst_rank, dst_off_f32, vals)

            for rank in 0..n {
                let op = &plans[rank].ops[op_idx];
                let bufs = &plans[rank].buffers;
                let oc = op.opcode;

                if oc == Opcode::Noop as u8 || oc == Opcode::Signal as u8
                    || oc == Opcode::Wait as u8 {
                    continue;
                }

                if oc == Opcode::Put as u8 || oc == Opcode::PutWithSignal as u8 {
                    let src = &bufs[op.src_buf as usize];
                    let sz = src.size as usize / 4;
                    let vals = read_buf(&data[rank], &scratch[rank], src, sz);
                    let dst = &bufs[op.dst_buf as usize];
                    puts.push((op.dst_rank as usize, dst.offset as usize / 4, vals));
                    continue;
                }

                if oc == Opcode::LocalReduce as u8 {
                    let src = &bufs[op.src_buf as usize];
                    let dst = &bufs[op.dst_buf as usize];
                    let sz = src.size as usize / 4;
                    let so = src.offset as usize / 4;
                    let do_ = dst.offset as usize / 4;
                    for j in 0..sz { data[rank][do_ + j] += scratch[rank][so + j]; }
                    continue;
                }

                if oc == Opcode::LocalCopy as u8 {
                    let src = &bufs[op.src_buf as usize];
                    let dst = &bufs[op.dst_buf as usize];
                    let sz = src.size as usize / 4;
                    let so = src.offset as usize / 4;
                    let do_ = dst.offset as usize / 4;
                    for j in 0..sz { data[rank][do_ + j] = scratch[rank][so + j]; }
                    continue;
                }

                if oc == Opcode::WaitReducePut as u8 {
                    // 1. Reduce: data[dst_buf] += scratch[src_buf]
                    let src = &bufs[op.src_buf as usize];
                    let dst = &bufs[op.dst_buf as usize];
                    let sz = src.size as usize / 4;
                    let so = src.offset as usize / 4;
                    let do_ = dst.offset as usize / 4;
                    for j in 0..sz { data[rank][do_ + j] += scratch[rank][so + j]; }
                    // 2. Put: data[dst_buf] -> scratch[dst_rank][_pad buf]
                    let put_dst = &bufs[op._pad as usize];
                    let vals = data[rank][do_..do_ + sz].to_vec();
                    puts.push((op.dst_rank as usize, put_dst.offset as usize / 4, vals));
                    continue;
                }

                if oc == Opcode::WaitReduceCopy as u8 {
                    // 1. Reduce: data[_pad] += scratch[src_buf]
                    let src = &bufs[op.src_buf as usize];
                    let red_dst = &bufs[op._pad as usize];
                    let sz = src.size as usize / 4;
                    let so = src.offset as usize / 4;
                    let rdo = red_dst.offset as usize / 4;
                    for j in 0..sz { data[rank][rdo + j] += scratch[rank][so + j]; }
                    // 2. Copy: data[_pad] -> data[dst_buf]
                    let cpy_dst = &bufs[op.dst_buf as usize];
                    let cdo = cpy_dst.offset as usize / 4;
                    for j in 0..sz { data[rank][cdo + j] = data[rank][rdo + j]; }
                    continue;
                }

                panic!("unknown opcode {}", oc);
            }

            // Phase 2: apply all puts to remote scratch
            for (dst_rank, dst_off, vals) in puts {
                for (j, v) in vals.into_iter().enumerate() {
                    scratch[dst_rank][dst_off + j] = v;
                }
            }
        }

        // Verify: all ranks should have [36.0; nf] (sum 1+2+...+8 = 36)
        for rank in 0..n {
            for i in 0..nf {
                assert!((data[rank][i] - 36.0).abs() < 1e-4,
                    "rank {} elem {}: got {}, expected 36.0", rank, i, data[rank][i]);
            }
        }
    }
}
