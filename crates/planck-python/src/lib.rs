// ==== Planck PyO3 Bindings ====
//
// Exposes Plan Compiler to Python for torchair graph pass integration.
// PlanCache is the primary bridge: Rust compile -> PyO3 -> Python -> C++ custom ops.

use planck_core::{
    plan::{Collective, CompileRequest, ExecutionPlan, ReduceOp},
    template::PlanTemplate,
    topo::Topology,
};
use pyo3::{prelude::*, types::PyBytes};
use std::{collections::HashMap, sync::Mutex};

// ==== PyPlanView ====

#[pyclass(frozen)]
struct PyPlanView {
    plan: ExecutionPlan,
}

#[pymethods]
impl PyPlanView {
    #[getter]
    fn num_ranks(&self) -> u16 { self.plan.header.num_ranks }

    #[getter]
    fn my_rank(&self) -> u16 { self.plan.header.my_rank }

    #[getter]
    fn num_ops(&self) -> u16 { self.plan.header.num_ops }

    #[getter]
    fn num_buffers(&self) -> u16 { self.plan.header.num_buffers }

    #[getter]
    fn num_streams(&self) -> u8 { self.plan.header.num_streams }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.plan.serialize())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPlanView(ranks={}, my_rank={}, ops={}, bufs={}, streams={})",
            self.plan.header.num_ranks,
            self.plan.header.my_rank,
            self.plan.header.num_ops,
            self.plan.header.num_buffers,
            self.plan.header.num_streams,
        )
    }
}

// ==== PyPlanTemplate ====

#[pyclass]
struct PyPlanTemplate {
    inner: PlanTemplate,
}

#[pymethods]
impl PyPlanTemplate {
    fn instantiate(&self, msg_size: usize) -> PyPlanView {
        PyPlanView { plan: self.inner.instantiate(msg_size) }
    }

    #[getter]
    fn num_ops(&self) -> usize { self.inner.frozen_ops.len() }

    #[getter]
    fn num_buffers(&self) -> usize { self.inner.buffer_exprs.len() }

    fn __repr__(&self) -> String {
        format!(
            "PyPlanTemplate(ops={}, bufs={}, base_msg={})",
            self.inner.frozen_ops.len(),
            self.inner.buffer_exprs.len(),
            self.inner.base_msg_size,
        )
    }
}

// ==== PlanCompiler ====

fn parse_reduce_op(s: &str) -> PyResult<ReduceOp> {
    match s.to_lowercase().as_str() {
        "sum" => Ok(ReduceOp::Sum),
        "max" => Ok(ReduceOp::Max),
        "min" => Ok(ReduceOp::Min),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown reduce_op '{}', expected sum/max/min",
            s
        ))),
    }
}

#[pyclass]
struct PlanCompiler {
    topo: Topology,
}

#[pymethods]
impl PlanCompiler {
    #[staticmethod]
    fn hccs_8card() -> Self { Self { topo: Topology::hccs_8card() } }

    #[pyo3(signature = (msg_size, my_rank, reduce_op="sum", pipeline_chunks=1))]
    fn compile_allreduce(
        &self,
        py: Python<'_>,
        msg_size: usize,
        my_rank: usize,
        reduce_op: &str,
        pipeline_chunks: usize,
    ) -> PyResult<PyPlanView> {
        let op = parse_reduce_op(reduce_op)?;
        let topo = &self.topo;
        let num_ranks = topo.num_ranks;

        let plan = py.allow_threads(|| {
            planck_core::plan::compile(
                &CompileRequest {
                    collective: Collective::AllReduce,
                    msg_size,
                    reduce_op: op,
                    num_ranks,
                    my_rank,
                    pipeline_chunks,
                },
                topo,
            )
        });
        Ok(PyPlanView { plan })
    }

    #[pyo3(signature = (my_rank, reduce_op="sum", pipeline_chunks=1, base_msg_size=8192))]
    fn compile_template(
        &self,
        py: Python<'_>,
        my_rank: usize,
        reduce_op: &str,
        pipeline_chunks: usize,
        base_msg_size: usize,
    ) -> PyResult<PyPlanTemplate> {
        let op = parse_reduce_op(reduce_op)?;
        let topo = &self.topo;
        let num_ranks = topo.num_ranks;

        let plan = py.allow_threads(|| {
            planck_core::plan::compile(
                &CompileRequest {
                    collective: Collective::AllReduce,
                    msg_size: base_msg_size,
                    reduce_op: op,
                    num_ranks,
                    my_rank,
                    pipeline_chunks,
                },
                topo,
            )
        });
        Ok(PyPlanTemplate { inner: PlanTemplate::from_plan(plan, base_msg_size) })
    }
}

// ==== PlanCache ====

#[pyclass]
struct PlanCache {
    compiler: PlanCompiler,
    cache: Mutex<HashMap<(usize, usize), ExecutionPlan>>,
}

// ExecutionPlan needs Clone for cache retrieval
fn clone_plan(p: &ExecutionPlan) -> ExecutionPlan {
    ExecutionPlan { header: p.header, buffers: p.buffers.clone(), ops: p.ops.clone() }
}

#[pymethods]
impl PlanCache {
    #[staticmethod]
    fn hccs_8card() -> Self {
        Self { compiler: PlanCompiler::hccs_8card(), cache: Mutex::new(HashMap::new()) }
    }

    #[pyo3(signature = (msg_size, my_rank, reduce_op="sum", pipeline_chunks=1))]
    fn get_allreduce(
        &self,
        py: Python<'_>,
        msg_size: usize,
        my_rank: usize,
        reduce_op: &str,
        pipeline_chunks: usize,
    ) -> PyResult<PyPlanView> {
        let key = (msg_size, my_rank);

        // Fast path: cache hit
        {
            let cache = self.cache.lock().unwrap();
            if let Some(plan) = cache.get(&key) {
                return Ok(PyPlanView { plan: clone_plan(plan) });
            }
        }

        // Slow path: compile (GIL released) then cache
        let view =
            self.compiler.compile_allreduce(py, msg_size, my_rank, reduce_op, pipeline_chunks)?;

        let mut cache = self.cache.lock().unwrap();
        cache.insert(key, clone_plan(&view.plan));
        Ok(view)
    }

    fn cache_size(&self) -> usize { self.cache.lock().unwrap().len() }

    fn clear(&self) { self.cache.lock().unwrap().clear(); }
}

// ==== Simulation ====

#[pyfunction]
#[pyo3(signature = (plans, config_toml=None))]
fn simulate(
    py: Python<'_>,
    plans: Vec<PyRef<'_, PyPlanView>>,
    config_toml: Option<&str>,
) -> PyResult<String> {
    use planck_core::sim;

    let cfg = match config_toml {
        Some(s) => sim::SimConfig::from_toml(s).map_err(pyo3::exceptions::PyValueError::new_err)?,
        None => sim::SimConfig::default(),
    };

    let exec_plans: Vec<_> = plans.iter().map(|p| p.plan.clone()).collect();
    let topo = Topology::hccs_8card();

    let trace = py.allow_threads(|| sim::simulate(&exec_plans, &topo, &cfg));
    Ok(trace.to_json())
}

// ==== Module ====

#[pymodule]
fn _planck(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyPlanView>()?;
    m.add_class::<PyPlanTemplate>()?;
    m.add_class::<PlanCompiler>()?;
    m.add_class::<PlanCache>()?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
