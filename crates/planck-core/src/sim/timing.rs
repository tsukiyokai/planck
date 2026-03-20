// ==== TimingModel: pluggable latency/bandwidth models ====
//
// SimpleModel: alpha-beta, one-line per method. For parameter sweeps.
// AscendModel: hardware-aware (3-round notify, GET mode, InlineReduce overlap).
// Source: HCOMM phase2-hcomm-platform, prim_rules.cc, dispatcher_pub.h.

use crate::topo::Link;
use super::config::SimConfig;

pub trait TimingModel {
    /// Data transfer time (us) for `size` bytes over `link`.
    fn put_time(&self, link: &Link, size: usize) -> f64;
    /// Signal/notify completion time (us).
    fn notify_time(&self, link: &Link) -> f64;
    /// Local reduce time (us), HBM-bandwidth bound.
    fn reduce_time(&self, size: usize) -> f64;
    /// Fused Wait+Reduce+Put: may overlap reduce and put.
    fn inline_reduce_put_time(&self, link: &Link, size: usize) -> f64;
    /// Kernel launch overhead between consecutive ops (us).
    fn kernel_launch_overhead(&self) -> f64;
}

// ==== SimpleModel: alpha-beta ====

pub struct SimpleModel {
    pub hbm_bw_gbps: f64,
}

impl TimingModel for SimpleModel {
    fn put_time(&self, link: &Link, size: usize) -> f64 {
        // latency + size / bandwidth
        link.latency_us + size as f64 / (link.bandwidth_gbps * 1e3)
    }
    fn notify_time(&self, link: &Link) -> f64 {
        link.latency_us
    }
    fn reduce_time(&self, size: usize) -> f64 {
        size as f64 / (self.hbm_bw_gbps * 1e3)
    }
    fn inline_reduce_put_time(&self, link: &Link, size: usize) -> f64 {
        self.put_time(link, size) + self.reduce_time(size) // no overlap
    }
    fn kernel_launch_overhead(&self) -> f64 { 0.0 }
}

// ==== AscendModel: HCCS hardware-aware ====

pub struct AscendModel {
    pub notify_rounds: u32,  // 3-round handshake (phase2-hcomm-platform)
    pub hbm_bw_gbps:   f64,  // HBM bandwidth
    pub launch_us:     f64,  // kernel launch overhead
}

impl TimingModel for AscendModel {
    fn put_time(&self, link: &Link, size: usize) -> f64 {
        // GET mode (prim_rules.cc): request + data = 2 * latency + transfer
        2.0 * link.latency_us + size as f64 / (link.bandwidth_gbps * 1e3)
    }

    fn notify_time(&self, link: &Link) -> f64 {
        self.notify_rounds as f64 * link.latency_us
    }

    fn reduce_time(&self, size: usize) -> f64 {
        size as f64 / (self.hbm_bw_gbps * 1e3)
    }

    fn inline_reduce_put_time(&self, link: &Link, size: usize) -> f64 {
        // MTE+AIV physical isolation (dispatcher_pub.h):
        // notify wait, then overlapped reduce+put
        let reduce = self.reduce_time(size);
        let put    = self.put_time(link, size);
        let notify = self.notify_time(link);
        notify + reduce.max(put)
    }

    fn kernel_launch_overhead(&self) -> f64 { self.launch_us }
}

// ==== Factory ====

pub fn create_model(cfg: &SimConfig) -> Box<dyn TimingModel> {
    match cfg.timing.model.as_str() {
        "simple" => Box::new(SimpleModel {
            hbm_bw_gbps: cfg.timing.hbm_bw_gbps,
        }),
        _ => Box::new(AscendModel {
            notify_rounds: cfg.timing.notify_rounds,
            hbm_bw_gbps:  cfg.timing.hbm_bw_gbps,
            launch_us:     5.0,
        }),
    }
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
    fn sim_simple_put_time() {
        let m = SimpleModel { hbm_bw_gbps: 460.0 };
        let t = m.put_time(&test_link(), 30_000); // 30KB
        assert!(t > 1.5);  // at least latency
        assert!(t < 10.0);
    }

    #[test]
    fn sim_ascend_notify_3_rounds() {
        let m = AscendModel { notify_rounds: 3, hbm_bw_gbps: 460.0, launch_us: 5.0 };
        let t = m.notify_time(&test_link());
        assert!((t - 4.5).abs() < 0.01); // 3 * 1.5us
    }

    #[test]
    fn sim_ascend_inline_reduce_overlaps() {
        let m = AscendModel { notify_rounds: 3, hbm_bw_gbps: 460.0, launch_us: 5.0 };
        let link = test_link();
        let fused    = m.inline_reduce_put_time(&link, 256_000);
        let separate = m.notify_time(&link) + m.reduce_time(256_000) + m.put_time(&link, 256_000);
        assert!(fused < separate, "InlineReduce should overlap: fused={fused:.3} separate={separate:.3}");
    }
}
