// ==== Alpha-Beta-Gamma Cost Model ====
//
// Hierarchical cost model. v0.1: single-level (HCCS only).
// T = latency_steps * alpha + volume * beta + compute_volume * gamma

use crate::topo::Topology;

pub struct CostModel {
    pub alpha_us:         f64, // startup latency per step (us)
    pub beta_us_per_byte: f64, // per-byte transfer time (us/byte)
    pub gamma_us_per_byte: f64, // per-byte compute time (us/byte)
}

impl CostModel {
    /// Extract cost parameters from topology's link properties.
    pub fn from_topology(topo: &Topology) -> Self {
        // Use first link's properties (uniform topology in v0.1)
        let link = &topo.links[0];
        Self {
            alpha_us:          link.latency_us,
            beta_us_per_byte:  1.0 / (link.bandwidth_gbps * 1e9 / 1e6), // us/byte = 1/(GB/s * 1e9 / 1e6)
            gamma_us_per_byte: 0.0, // ignore compute cost for v0.1
        }
    }

    /// Ring AllReduce cost:
    ///   T = 2*(n-1)*alpha + 2*(n-1)/n * M * beta + (n-1)/n * M * gamma
    pub fn ring_allreduce(&self, msg_size: usize, num_ranks: usize) -> f64 {
        let n = num_ranks as f64;
        let m = msg_size as f64;
        2.0 * (n - 1.0) * self.alpha_us
            + 2.0 * (n - 1.0) / n * m * self.beta_us_per_byte
            + (n - 1.0) / n * m * self.gamma_us_per_byte
    }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::Topology;

    #[test]
    fn cost_from_topology() {
        let topo = Topology::hccs_8card();
        let cost = CostModel::from_topology(&topo);
        assert!(cost.alpha_us > 0.0);
        assert!(cost.beta_us_per_byte > 0.0);
    }

    #[test]
    fn ring_allreduce_cost_scales_with_size() {
        let topo = Topology::hccs_8card();
        let cost = CostModel::from_topology(&topo);
        let t_small = cost.ring_allreduce(1024, 8);
        let t_large = cost.ring_allreduce(256 << 20, 8);
        assert!(t_large > t_small);
        // At 30 GB/s HCCS, alpha dominates small msgs; ratio ~743
        assert!(t_large / t_small > 100.0);
    }

    #[test]
    fn ring_cost_formula() {
        let cost = CostModel {
            alpha_us:          10.0,
            beta_us_per_byte:  0.001,
            gamma_us_per_byte: 0.0,
        };
        let t = cost.ring_allreduce(8000, 8);
        let expected = 2.0 * 7.0 * 10.0 + 2.0 * 7.0 / 8.0 * 8000.0 * 0.001;
        assert!((t - expected).abs() < 1e-6);
    }
}
