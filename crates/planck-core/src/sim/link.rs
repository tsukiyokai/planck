// ==== LinkState: bandwidth competition model ====
//
// Fair-share model: effective_bw = link_bw / active_flows.
// Sufficient for analytical-level accuracy (Echo paper: ~8% error).

use crate::topo::Link;

pub struct LinkState {
    pub link:         Link,
    pub active_flows: u32,
}

impl LinkState {
    pub fn new(link: Link) -> Self {
        Self { link, active_flows: 0 }
    }

    /// Per-flow bandwidth under fair sharing (GB/s).
    pub fn effective_bw_gbps(&self) -> f64 {
        if self.active_flows == 0 {
            self.link.bandwidth_gbps
        } else {
            self.link.bandwidth_gbps / self.active_flows as f64
        }
    }

    pub fn add_flow(&mut self)    { self.active_flows += 1; }
    pub fn remove_flow(&mut self) { self.active_flows = self.active_flows.saturating_sub(1); }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::TransportType;

    fn test_link() -> Link {
        Link { src: 0, dst: 1, bandwidth_gbps: 30.0, latency_us: 1.5, transport: TransportType::Hccs }
    }

    #[test]
    fn sim_link_no_contention() {
        let ls = LinkState::new(test_link());
        assert_eq!(ls.effective_bw_gbps(), 30.0);
    }

    #[test]
    fn sim_link_fair_share() {
        let mut ls = LinkState::new(test_link());
        ls.add_flow();
        ls.add_flow();
        assert!((ls.effective_bw_gbps() - 15.0).abs() < 0.01);
        ls.remove_flow();
        assert!((ls.effective_bw_gbps() - 30.0).abs() < 0.01);
    }
}
