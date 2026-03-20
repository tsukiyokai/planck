// ==== planck-sim: embedded DES simulator for plan schedule verification ====
//
// Not a general CCL simulator. An embedded compiler feedback tool.
// Answers "is this schedule good?" not "how fast on real hardware?"

pub mod config;
pub mod engine;
pub mod link;
pub mod timing;
pub mod trace;

pub use config::SimConfig;
pub use trace::Trace;

use crate::plan::ExecutionPlan;
use crate::topo::Topology;

/// Simulate execution of plans for all ranks, return Chrome Trace.
pub fn simulate(plans: &[ExecutionPlan], topo: &Topology, cfg: &SimConfig) -> Trace {
    let model = timing::create_model(cfg);
    let mut sim = engine::Simulator::new(plans, topo, model, cfg);
    sim.run();
    sim.into_trace()
}

// ==== Integration Tests ====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Collective, CompileRequest, ReduceOp, compile};

    fn compile_plans(msg_size: usize, chunks: usize) -> (Vec<ExecutionPlan>, Topology) {
        let topo = Topology::hccs_8card();
        let plans: Vec<_> = (0..8).map(|r| compile(&CompileRequest {
            collective: Collective::AllReduce,
            msg_size, reduce_op: ReduceOp::Sum,
            num_ranks: 8, my_rank: r, pipeline_chunks: chunks,
        }, &topo)).collect();
        (plans, topo)
    }

    #[test]
    fn sim_trace_has_events() {
        let cfg = SimConfig::default();
        let (plans, topo) = compile_plans(256, 1);
        let trace = simulate(&plans, &topo, &cfg);
        assert!(!trace.events.is_empty(), "trace should have events");
        let json = trace.to_json();
        assert!(json.contains("traceEvents"));
    }

    #[test]
    fn sim_monotonic_with_size() {
        let cfg = SimConfig::default();
        let (small, topo) = compile_plans(256, 1);
        let (large, _)    = compile_plans(256_000, 1);
        let ts = simulate(&small, &topo, &cfg).total_time();
        let tl = simulate(&large, &topo, &cfg).total_time();
        assert!(tl > ts, "larger msg ({tl:.1}us) should take longer than small ({ts:.1}us)");
    }

    #[test]
    fn sim_pipeline_overlap() {
        let cfg = SimConfig::default();
        let (plans4, topo) = compile_plans(256_000, 4);
        let (plans1, _)    = compile_plans(256_000, 1);
        let t4 = simulate(&plans4, &topo, &cfg).total_time();
        let t1 = simulate(&plans1, &topo, &cfg).total_time();
        assert!(t4 < t1, "4-chunk ({t4:.1}us) should be faster than 1-chunk ({t1:.1}us)");
    }

    #[test]
    fn sim_completes_without_deadlock() {
        // Verify all 8 ranks complete (no deadlock from signal/wait mismatch)
        let cfg = SimConfig::default();
        let (plans, topo) = compile_plans(8192, 2);
        let trace = simulate(&plans, &topo, &cfg);
        assert!(trace.total_time() > 0.0, "simulation should produce positive time");
        // Each rank should produce trace events
        for r in 0..8u16 {
            let rank_events = trace.events.iter().filter(|e| e.pid == r).count();
            assert!(rank_events > 0, "rank {r} should have trace events");
        }
    }
}
