// ==== Parameterized Plan Templates ====
//
// For inference: plan structure (ops, deps) is constant across message sizes.
// Only buffer offsets/sizes scale. Template freezes ops, parameterizes buffers.
// Instantiation is O(num_buffers), sub-microsecond.

use crate::plan::{BufEntry, ExecutionPlan, OpEntry, PlanHeader};

// ==== Types ====

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamSlot {
    MsgSize,
}

#[derive(Clone, Debug)]
pub struct BufExpr {
    pub pool:         u32,
    pub offset_scale: f64, // offset = offset_scale * msg_size
    pub size_scale:   f64, // size   = size_scale   * msg_size
}

pub struct PlanTemplate {
    pub frozen_ops:    Vec<OpEntry>,
    pub buffer_exprs:  Vec<BufExpr>,
    pub header:        PlanHeader,
    pub param_slots:   Vec<ParamSlot>,
    pub base_msg_size: usize,
}

// ==== Impl ====

impl PlanTemplate {
    /// Freeze a compiled plan into a template. Buffer sizes become linear functions of msg_size.
    pub fn from_plan(plan: ExecutionPlan, base_msg_size: usize) -> Self {
        let m = base_msg_size as f64;
        let buffer_exprs = plan.buffers.iter().map(|b| BufExpr {
            pool:         b.pool,
            offset_scale: b.offset as f64 / m,
            size_scale:   b.size as f64 / m,
        }).collect();

        Self {
            frozen_ops: plan.ops,
            buffer_exprs,
            header: plan.header,
            param_slots: vec![ParamSlot::MsgSize],
            base_msg_size,
        }
    }

    /// Instantiate template with a new msg_size. O(num_buffers).
    pub fn instantiate(&self, msg_size: usize) -> ExecutionPlan {
        let m = msg_size as f64;
        let buffers: Vec<BufEntry> = self.buffer_exprs.iter().map(|e| BufEntry {
            pool:   e.pool,
            offset: (e.offset_scale * m) as u32,
            size:   (e.size_scale * m) as u32,
        }).collect();

        let mut header = self.header;
        header.num_buffers = buffers.len() as u16;

        ExecutionPlan {
            header,
            buffers,
            ops: self.frozen_ops.clone(),
        }
    }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{CompileRequest, Collective, ReduceOp, compile};
    use crate::topo::Topology;

    fn make_template() -> PlanTemplate {
        let plan = compile(
            &CompileRequest {
                collective:      Collective::AllReduce,
                msg_size:        8192,
                reduce_op:       ReduceOp::Sum,
                num_ranks:       8,
                my_rank:         0,
                pipeline_chunks: 1,
            },
            &Topology::hccs_8card(),
        );
        PlanTemplate::from_plan(plan, 8192)
    }

    #[test]
    fn template_creation() {
        let t = make_template();
        assert_eq!(t.param_slots.len(), 1);
        assert!(!t.frozen_ops.is_empty());
        assert!(!t.buffer_exprs.is_empty());
    }

    #[test]
    fn template_instantiation() {
        let t = make_template();
        let p2k = t.instantiate(2048);
        let p4k = t.instantiate(4096);
        // Same ops, different buffer sizes
        assert_eq!(p2k.ops.len(), p4k.ops.len());
        assert!(p2k.buffers[0].size < p4k.buffers[0].size);
    }

    #[test]
    fn instantiation_is_fast() {
        let t = make_template();
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(t.instantiate(1024));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 10, "1000 took {}ms", elapsed.as_millis());
    }
}
