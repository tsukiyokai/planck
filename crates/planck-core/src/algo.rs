// ==== Ring AllReduce Algorithm Decomposition ====
//
// Decomposes AllReduce into ReduceScatter + AllGather steps.
// Output: per-rank list of AlgoStep (abstract, before instruction generation).
// Ring order: 0 -> 1 -> 2 -> ... -> n-1 -> 0

// ==== Types ====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    ReduceScatter,
    AllGather,
}

#[derive(Debug, Clone)]
pub struct AlgoStep {
    pub phase:      Phase,
    pub step:       u16,   // step index within phase (0..n-2)
    pub send_chunk: u16,   // data chunk index to send
    pub recv_chunk: u16,   // data chunk index to receive
    pub dst_rank:   u16,   // send to
    pub src_rank:   u16,   // receive from
    pub needs_reduce: bool,
}

// ==== Ring AllReduce ====

/// Generate Ring AllReduce steps for given rank.
///
/// Ring: send to (rank+1)%n, recv from (rank-1+n)%n.
/// n-1 ReduceScatter steps + n-1 AllGather steps = 2*(n-1) total.
pub fn ring_allreduce(num_ranks: u16, my_rank: u16) -> Vec<AlgoStep> {
    let n = num_ranks as i32;
    let r = my_rank as i32;
    let dst = ((r + 1) % n) as u16;
    let src = ((r - 1 + n) % n) as u16;
    let steps = (n - 1) as u16;

    let mut result = Vec::with_capacity(2 * steps as usize);

    // ReduceScatter: step k sends chunk (r-k+n)%n, recvs chunk (r-k-1+n)%n
    for k in 0..steps {
        let ki = k as i32;
        result.push(AlgoStep {
            phase:      Phase::ReduceScatter,
            step:       k,
            send_chunk: ((r - ki + n) % n) as u16,
            recv_chunk: ((r - ki - 1 + n) % n) as u16,
            dst_rank:   dst,
            src_rank:   src,
            needs_reduce: true,
        });
    }

    // AllGather: step k sends chunk (r-k+1+n)%n, recvs chunk (r-k+n)%n
    for k in 0..steps {
        let ki = k as i32;
        result.push(AlgoStep {
            phase:      Phase::AllGather,
            step:       k,
            send_chunk: ((r - ki + 1 + n) % n) as u16,
            recv_chunk: ((r - ki + n) % n) as u16,
            dst_rank:   dst,
            src_rank:   src,
            needs_reduce: false,
        });
    }

    result
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_allreduce_step_count() {
        let steps = ring_allreduce(8, 0);
        // RS: 7 steps + AG: 7 steps = 14
        assert_eq!(steps.len(), 14);
        assert_eq!(steps.iter().filter(|s| s.phase == Phase::ReduceScatter).count(), 7);
        assert_eq!(steps.iter().filter(|s| s.phase == Phase::AllGather).count(), 7);
    }

    #[test]
    fn ring_send_recv_ranks() {
        let steps = ring_allreduce(8, 3); // rank 3
        for step in &steps {
            assert_eq!(step.dst_rank, 4); // (3+1)%8
            assert_eq!(step.src_rank, 2); // (3-1+8)%8
        }
    }

    #[test]
    fn ring_rs_chunk_indices() {
        let steps = ring_allreduce(8, 0);
        let rs_steps: Vec<_> = steps.iter()
            .filter(|s| s.phase == Phase::ReduceScatter)
            .collect();
        // Rank 0, RS step 0: send chunk (0-0+8)%8=0, recv chunk (0-0-1+8)%8=7
        assert_eq!(rs_steps[0].send_chunk, 0);
        assert_eq!(rs_steps[0].recv_chunk, 7);
    }

    #[test]
    fn all_chunks_covered() {
        for rank in 0..8u16 {
            let steps = ring_allreduce(8, rank);
            let rs_recv: std::collections::HashSet<_> = steps.iter()
                .filter(|s| s.phase == Phase::ReduceScatter)
                .map(|s| s.recv_chunk)
                .collect();
            // Rank receives 7 distinct chunks (all except its own final chunk)
            assert_eq!(rs_recv.len(), 7);
        }
    }
}
