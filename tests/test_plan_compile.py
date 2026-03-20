"""Planck PyO3 bindings tests — validates Rust Plan Compiler exposed to Python."""

import planck


def test_import():
    """Module loads and exposes __version__."""
    assert hasattr(planck, "__version__")
    assert isinstance(planck.__version__, str)
    assert len(planck.__version__) > 0


def test_compile_allreduce():
    """Compile a 256MB AllReduce plan and check fields."""
    compiler = planck.PlanCompiler.hccs_8card()
    view = compiler.compile_allreduce(256 << 20, my_rank=0, pipeline_chunks=4)

    assert view.num_ranks == 8
    assert view.my_rank == 0
    assert view.num_ops > 0
    assert view.num_buffers > 0
    assert view.num_streams == 4  # 4 pipeline chunks -> 4 streams

    raw = view.to_bytes()
    assert isinstance(raw, bytes)
    assert len(raw) > 32  # at least header


def test_plan_cache():
    """Second call with same key hits cache."""
    cache = planck.PlanCache.hccs_8card()
    assert cache.cache_size() == 0

    v1 = cache.get_allreduce(1 << 20, my_rank=3)
    assert cache.cache_size() == 1

    v2 = cache.get_allreduce(1 << 20, my_rank=3)
    assert cache.cache_size() == 1  # no new entry

    assert v1.num_ops == v2.num_ops
    assert v1.num_buffers == v2.num_buffers
    assert v1.to_bytes() == v2.to_bytes()


def test_template_instantiate():
    """Template produces same ops but different buffer sizes."""
    compiler = planck.PlanCompiler.hccs_8card()
    tmpl = compiler.compile_template(my_rank=0, pipeline_chunks=1)

    p2k = tmpl.instantiate(2048)
    p4k = tmpl.instantiate(4096)

    assert p2k.num_ops == p4k.num_ops       # same op structure
    assert p2k.num_ops == tmpl.num_ops
    assert p2k.to_bytes() != p4k.to_bytes()  # different buffer sizes
