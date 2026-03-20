// ==== 8-Card HCCS Topology ====
//
// v0.1: hardcoded all-to-all 8-card topology.
// Atlas 800T A2: 56 GB/s per HCCS port, 3-4 ports per die.
// Conservative per-link estimate: 30 GB/s.

// ==== Calibratable Constants ====

const HCCS_BW_GBPS: f64 = 30.0;  // GB/s per link (conservative)
const HCCS_LAT_US: f64  = 1.5;   // microseconds per hop

// ==== Types ====

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    Hccs = 0,
    Roce,
    Shm,
}

#[derive(Debug, Clone)]
pub struct Link {
    pub src:           usize,
    pub dst:           usize,
    pub bandwidth_gbps: f64,
    pub latency_us:    f64,
    pub transport:     TransportType,
}

#[derive(Debug, Clone)]
pub struct Topology {
    pub num_ranks: usize,
    pub links:     Vec<Link>,
}

impl Topology {
    /// 8-card HCCS: full mesh (all-to-all), 56 directed links.
    pub fn hccs_8card() -> Self {
        let n = 8;
        let mut links = Vec::with_capacity(n * (n - 1));
        for src in 0..n {
            for dst in 0..n {
                if src != dst {
                    links.push(Link {
                        src,
                        dst,
                        bandwidth_gbps: HCCS_BW_GBPS,
                        latency_us:     HCCS_LAT_US,
                        transport:      TransportType::Hccs,
                    });
                }
            }
        }
        Self { num_ranks: n, links }
    }

    pub fn has_link(&self, src: usize, dst: usize) -> bool {
        self.links.iter().any(|l| l.src == src && l.dst == dst)
    }

    pub fn get_link(&self, src: usize, dst: usize) -> Option<&Link> {
        self.links.iter().find(|l| l.src == src && l.dst == dst)
    }

    /// Simple sequential ring for v0.1: [0, 1, 2, ..., n-1].
    pub fn ring_order(&self) -> Vec<usize> {
        (0..self.num_ranks).collect()
    }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hccs_8card_basics() {
        let topo = Topology::hccs_8card();
        assert_eq!(topo.num_ranks, 8);
        // 8 cards all-to-all: 8*7 = 56 directed links
        assert_eq!(topo.links.len(), 56);
    }

    #[test]
    fn hccs_8card_ring_neighbors() {
        let topo = Topology::hccs_8card();
        let ring = topo.ring_order();
        assert_eq!(ring.len(), 8);
        for i in 0..8 {
            let next = ring[(i + 1) % 8];
            assert!(topo.has_link(ring[i], next));
        }
    }

    #[test]
    fn link_properties() {
        let topo = Topology::hccs_8card();
        let link = topo.get_link(0, 1).unwrap();
        assert_eq!(link.transport, TransportType::Hccs);
        assert!(link.bandwidth_gbps > 0.0);
        assert!(link.latency_us > 0.0);
    }
}
