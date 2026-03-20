"""planck-sim integration test via Python."""
import json
import planck


def test_simulate_returns_json():
    compiler = planck.PlanCompiler.hccs_8card()
    plans = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=1) for r in range(8)]
    trace_json = planck.simulate(plans)
    data = json.loads(trace_json)
    assert "traceEvents" in data
    assert len(data["traceEvents"]) > 0


def test_simulate_with_config():
    config = """
[timing]
model = "simple"
"""
    compiler = planck.PlanCompiler.hccs_8card()
    plans = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=4) for r in range(8)]
    trace_json = planck.simulate(plans, config_toml=config)
    data = json.loads(trace_json)
    assert len(data["traceEvents"]) > 0


def test_pipeline_more_events():
    compiler = planck.PlanCompiler.hccs_8card()
    plans1 = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=1) for r in range(8)]
    plans4 = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=4) for r in range(8)]
    t1 = json.loads(planck.simulate(plans1))
    t4 = json.loads(planck.simulate(plans4))
    # 4-chunk pipeline produces more ops (more fine-grained schedule)
    assert len(t4["traceEvents"]) >= len(t1["traceEvents"])
