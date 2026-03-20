# Planck v0.1 Implementation -- Distilled Execution Plan

> 精读 `docs/plans/2026-03-19-planck-v01-implementation.md` 后的结构化提炼。
> 聚焦 Chunks 1-5 (Phase A, Rust + Python)。

## Status

Phase A complete. 29/29 Rust tests, 4/4 Python tests, 0 warnings.
Benchmark: compile ~1.36us (红线 <1ms), instantiate ~73ns (红线 <1us).

---

## File Structure

```
planck/
  Cargo.toml                              workspace root (default-members=planck-core)
  pyproject.toml                           maturin config -> planck._planck
  rust-toolchain.toml                      pin stable

  crates/planck-core/                      ZERO external deps (v0.1)
    src/lib.rs                             re-exports 6 modules
    src/plan.rs         683L               IR types + serialize + fuse + compile
    src/topo.rs         106L               8-card HCCS topology
    src/cost.rs          75L               alpha-beta-gamma cost model
    src/algo.rs         120L               Ring AllReduce decomposition
    src/sched.rs        209L               pipeline scheduler + double-buf
    src/template.rs     124L               parameterized plan templates
    benches/compile_bench.rs  51L          criterion guards

  crates/planck-python/                    PyO3 cdylib
    src/lib.rs          249L               PlanCompiler/PlanCache/PyPlanView/PyPlanTemplate

  python/planck/
    __init__.py                            re-exports from _planck

  tests/
    test_plan_compile.py                   4 pytest tests

  csrc/                                    Phase B (not yet implemented)
    include/planck/{plan,transport,executor}.h
    transport/{hccs,mock}.cpp
    ops/{pipelined_allreduce,quantized_allreduce,kv_pipeline_transfer}.cpp
    kernels/{quantize_per_group,reduce_add}.cpp
    executor/engine.cpp
    torch_binding.cpp
    CMakeLists.txt
```

---

## Dependency Graph

```
Chunk 1: Skeleton
    |   Cargo workspace + maturin + empty modules
    v
Chunk 2: IR + Topo + Cost                   [plan.rs, topo.rs, cost.rs]
    |   repr(C) structs, serialize, 8-card mesh, alpha-beta
    v
Chunk 3: Algo + Sched + Fusion              [algo.rs, sched.rs, plan.rs]
    |   Ring decomp, pipeline double-buf, 3 fusion patterns, compile()
    v
Chunk 4: E2E Sim + Template + Bench         [plan.rs, template.rs, compile_bench.rs]
    |   8-rank simulation, param slots, criterion guards
    v
Chunk 5: PyO3 Bindings                      [planck-python/src/lib.rs]
    |   PlanCompiler, PlanCache, GIL release
    |
    +----- Phase A boundary (macOS, no Ascend) -----
    |
    v
Chunk 6: C++ Execution Layer                [csrc/]
    |   plan.h mirrors Rust, transport abstract, executor, mock
    v
Chunk 7: torchair + Benchmarks              [python/planck/graph_pass.py, tests/bench_vs_hccl.py]
        pattern replacement, FakeTensor, 3 benchmark groups
```

依赖方向: 严格线性 1->2->3->4->5 | 6->7。Chunk 5的PyO3只依赖planck-core的public API。

---

## Per-Task Breakdown: Chunks 1-5

### Chunk 1: Project Skeleton (Task 1)

Goal: `cargo build` zero warnings, maturin config ready.

Key decisions:
- edition 2021, resolver "2", lto="fat" (release)
- planck-python: lib name="_planck", crate-type=["cdylib"]
- planck-core: zero external deps (only criterion dev-dep)
- maturin: python-source="python", module-name="planck._planck"

Files: Cargo.toml, rust-toolchain.toml, 2 crate Cargo.toml, pyproject.toml, 6 empty .rs modules

---

### Chunk 2: Plan IR + Topology + Cost (Tasks 2-4)

#### Task 2: Plan IR Types (plan.rs)

Key types (all `repr(C)`, for zero-copy Rust-to-C++ FFI):

| Struct       | Size  | Fields                                                            |
|:-------------|:------|:------------------------------------------------------------------|
| `PlanHeader` | 32B   | magic(u32) version(u16) num_ops/buffers(u16) streams/events(u8)  |
|              |       | num_ranks/my_rank(u16) flags(u32) _reserved([u8;12])             |
| `BufEntry`   | 12B   | pool(u32) offset(u32) size(u32)                                  |
| `OpEntry`    | 16B   | opcode/stream_id/reduce_op/flags(u8x4)                           |
|              |       | src_buf/dst_buf/dst_rank/wait_event/signal_event/_pad (u16x6)    |

Enums:
- `Opcode`: Put=0 Signal Wait LocalCopy LocalReduce PutWithSignal WaitReduceCopy WaitReducePut Noop=8
- `ReduceOp`: Sum=0 Max Min
- `BufPool`: Scratch=0 Input Output

Constants: `PLAN_MAGIC=0x4B4E_4C50` ("PLNK"), `PLAN_VERSION=1`

Methods: `PlanHeader::new()`, `OpEntry::new()`, `ExecutionPlan::serialize()` / `deserialize()`

Serialization: `header_bytes ++ buf_entries_bytes ++ op_entries_bytes` (raw repr(C) memcpy)

Tests (6):
1. `header_is_32_bytes` -- size_of check
2. `buf_entry_is_12_bytes`
3. `op_entry_is_16_bytes`
4. `header_magic` -- new() sets PLAN_MAGIC + PLAN_VERSION
5. `opcode_values` -- Put=0, WaitReducePut=7, Noop=8
6. `serialize_roundtrip` -- serialize -> deserialize -> field equality

#### Task 3: Topology (topo.rs)

Key types:
- `TransportType` repr(u8): Hccs=0, Roce, Shm
- `Link`: src, dst, bandwidth_gbps(f64), latency_us(f64), transport
- `Topology`: num_ranks(usize), links(Vec<Link>)

Calibratable constants: `HCCS_BW_GBPS=30.0` GB/s, `HCCS_LAT_US=1.5` us

Methods:
- `Topology::hccs_8card()` -- 56 directed links (8x7 all-to-all)
- `has_link(src, dst)` / `get_link(src, dst)`
- `ring_order()` -> [0,1,2,...,7]

Tests (3+1):
1. `hccs_8card_basics` -- num_ranks=8, links.len()=56
2. `hccs_8card_ring_neighbors` -- ring traversal has_link
3. `link_properties` -- transport=Hccs, bw>0, latency>0

#### Task 4: Cost Model (cost.rs)

Key types:
- `CostModel { alpha_us, beta_us_per_byte, gamma_us_per_byte }`

Methods:
- `from_topology(topo)` -- alpha=link.latency, beta=1/(bw*1e3), gamma=0.0 (v0.1)
- `ring_allreduce(msg_size, num_ranks) -> f64`
  Formula: `T = 2(n-1)*alpha + 2(n-1)/n * M * beta + (n-1)/n * M * gamma`

Tests (3):
1. `cost_from_topology` -- alpha>0, beta>0
2. `ring_allreduce_cost_scales_with_size` -- 256MB/1KB ratio >100
3. `ring_cost_formula` -- exact formula validation with known values

---

### Chunk 3: Algo + Scheduler + Fusion (Tasks 5-7)

#### Task 5: Ring AllReduce Algorithm (algo.rs)

Key types:
- `Phase`: ReduceScatter, AllGather
- `AlgoStep { phase, step(u16), send_chunk(u16), recv_chunk(u16), dst_rank(u16), src_rank(u16), needs_reduce(bool) }`

Function: `ring_allreduce(num_ranks: u16, my_rank: u16) -> Vec<AlgoStep>`

Ring chunk index formulas:
- Ring: send to `(rank+1)%n`, recv from `(rank-1+n)%n`
- RS step k: send `(r-k+n)%n`, recv `(r-k-1+n)%n`, needs_reduce=true
- AG step k: send `(r-k+1+n)%n`, recv `(r-k+n)%n`, needs_reduce=false
- Total: 2*(n-1) steps

Tests (4):
1. `ring_allreduce_step_count` -- 14 steps for 8 ranks (7 RS + 7 AG)
2. `ring_send_recv_ranks` -- rank 3: dst=4, src=2
3. `ring_rs_chunk_indices` -- rank 0 RS step 0: send=0, recv=7
4. `all_chunks_covered` -- each rank recv 7 distinct chunks in RS

#### Task 6: Pipeline Scheduler (sched.rs)

Key types:
- `ScheduleResult { ops: Vec<OpEntry>, buffers: Vec<BufEntry>, num_streams: u8 }`

Function: `schedule(steps, msg_size, pipeline_chunks) -> ScheduleResult`

Buffer layout per pipeline chunk (N=num_ranks):
- N input sub-buffers (BufPool::Input, offset within user tensor, ring_piece each)
- N output sub-buffers (BufPool::Output, same offsets)
- 2 scratch buffers (double-buffered receive, ring_piece each)
- ring_piece = msg_size / (pipeline_chunks * num_ranks)

Op generation per stream c, per AlgoStep:
- RS step: Put(input[send_chunk] -> scratch) + Signal + Wait + LocalReduce(scratch -> input[recv_chunk])
- AG step: Put(input[send_chunk] -> scratch) + Signal + Wait + LocalCopy(scratch -> output[recv_chunk])
- stream_id = pipeline_chunk_index
- Scratch alternates a/b: `recv_buf = if step_idx % 2 == 0 { scratch_a } else { scratch_b }`

Tests (4):
1. `schedule_produces_ops` -- non-empty ops and buffers
2. `schedule_uses_c_streams` -- max stream = chunks-1
3. `schedule_buffer_sizes` -- scratch bufs >= C*2, each = ring_piece
4. `schedule_op_sequence_per_stream` -- 1 chunk: 14*4=56 ops on stream 0

#### Task 7: Fusion + compile() (plan.rs additions)

New types:
- `Collective { AllReduce }`
- `CompileRequest { collective, msg_size, reduce_op, num_ranks, my_rank, pipeline_chunks }`

Fusion function: `fuse(ops: Vec<OpEntry>) -> Vec<OpEntry>`
- Sliding window, greedy longest-match-first:
  - 3-op: Wait + LocalReduce + Put -> WaitReducePut (\_pad stores put's remote dst_buf)
  - 3-op: Wait + LocalReduce + LocalCopy -> WaitReduceCopy (\_pad stores reduce's dst_buf)
  - 2-op: Put + Signal -> PutWithSignal
- Must match on same stream_id

Compile function: `compile(req, topo) -> ExecutionPlan`
- Pipeline: topo -> cost -> algo(ring_allreduce) -> sched(schedule) -> fuse -> ExecutionPlan
- Event assignment: all 0 in v0.1 (independent pipeline chunks)

Tests (4):
1. `fusion_put_signal` -- 2 ops -> 1 PutWithSignal
2. `fusion_wait_reduce_put` -- 3 ops -> 1 WaitReducePut
3. `fusion_preserves_unfusable` -- Noop+Put stays 2
4. `compile_produces_valid_plan` -- magic, ranks, ops>0, ops < 56*4

---

### Chunk 4: E2E Simulation + Templates + Benchmarks (Tasks 8-10)

#### Task 8: End-to-End Simulation (plan.rs)

Two simulation tests (correctness gate for entire compiler):

1. `simulate_ring_allreduce` -- algorithm-level lockstep:
   - Compile 8 ranks, execute AlgoSteps directly
   - Input: rank r has [(r+1); 64], expect all [36.0; 64]
   - Validates: algo.rs chunk indexing correctness

2. `simulate_plan_execution` -- plan-level lockstep:
   - Compile 8 plans, execute OpEntry instructions (including fused ops)
   - Two-phase per op_idx: Phase 1 = local ops + stage puts, Phase 2 = apply puts to remote scratch
   - Validates: full pipeline algo -> sched -> fuse -> execution

Critical invariant: puts must be staged then applied (not interleaved), to prevent data races when
rank A's put writes to rank B's scratch while rank B reads it.

#### Task 9: Parameterized Templates (template.rs)

Key types:
- `ParamSlot { MsgSize }` (v0.1 only)
- `BufExpr { pool(u32), offset_scale(f64), size_scale(f64) }`
- `PlanTemplate { frozen_ops, buffer_exprs, header, param_slots, base_msg_size }`

Methods:
- `from_plan(plan, base_msg_size)` -- freeze ops, compute scales = field / msg_size
- `instantiate(msg_size)` -- evaluate scales * new_msg_size, O(num_buffers)

Design: ops are structural (which buf talks to which), buffer sizes are parametric.
Linear scaling assumption: offset = offset_scale * msg_size, size = size_scale * msg_size.

Tests (3):
1. `template_creation` -- param_slots.len()=1, frozen_ops non-empty
2. `template_instantiation` -- 2k vs 4k: same ops, different buffer sizes
3. `instantiation_is_fast` -- 1000 instantiations < 10ms

#### Task 10: Criterion Benchmarks (compile_bench.rs)

3 compile configs + 1 instantiate:
- `compile_256mb_4chunk`, `compile_16kb_1chunk`, `compile_1mb_2chunk` -- all must <1ms
- `instantiate_16kb` -- must <1us

Actual: compile ~1.36us, instantiate ~73ns. 3 orders of magnitude under redline.

---

### Chunk 5: PyO3 Bindings (Task 11)

Python classes (via PyO3):

| Class             | Key Methods/Properties                                           |
|:------------------|:-----------------------------------------------------------------|
| `PlanCompiler`    | `.hccs_8card()`, `.compile_allreduce(msg_size,rank,op,chunks)`, `.compile_template()` |
| `PlanCache`       | `.hccs_8card()`, `.get_allreduce()` (compile or cache hit), `.cache_size()`, `.clear()` |
| `PyPlanView`      | `.num_ranks`, `.my_rank`, `.num_ops`, `.num_buffers`, `.num_streams`, `.to_bytes()` |
| `PyPlanTemplate`  | `.instantiate(msg_size)`, `.num_ops`, `.num_buffers`             |

Implementation details:
- GIL released during compilation: `py.allow_threads(|| compile(...))`
- Cache key: (msg_size, my_rank), HashMap<(usize,usize), ExecutionPlan>
- `PyPlanView` is `#[pyclass(frozen)]` (immutable)
- `to_bytes()` returns raw serialized plan (passable to C++ executor)

Python tests (4):
1. `test_import` -- `planck.__version__` exists
2. `test_compile_allreduce` -- 256MB plan has expected fields + to_bytes() > 32
3. `test_plan_cache` -- second call hits cache, cache_size stays 1
4. `test_template_instantiate` -- same ops, different bytes

Build: `maturin develop` then `pytest tests/test_plan_compile.py`

---

## Key Findings (Decision Value)

1. Fusion \_pad field复用: WaitReducePut的\_pad存储put的remote dst_buf index。C++ executor (Chunk 6)必须对应读取\_pad来resolve put目标。这是Rust-C++ FFI的隐式契约。

2. Plan-level simulation的two-phase执行: 同一op_idx被所有rank同时执行时，必须先完成local ops再apply puts。这个invariant在C++ executor中需要用stream barrier保证(或者由one-sided transport的put/signal/wait协议保证)。

3. Cost model的gamma=0.0 (v0.1): compute cost被忽略。当引入quantized AllReduce (INT8 compress/decompress)时，gamma将变成非零，影响algorithm selection。

4. Template的linear scaling假设: offset_scale = base_offset / base_msg_size。当msg_size不是num_ranks的倍数时会产生截断误差。v0.1只测试对齐大小。

5. 环境陷阱: macOS上`cargo build`不包含planck-python (default-members限制)。PyO3构建必须用`maturin develop`。pytest需用miniconda Python。

---

## Phase B Preview (Chunks 6-7, requires Ascend NPU)

Chunk 6 (C++ Execution Layer): 6 tasks
- plan.h: pragma pack(1) mirrors Rust structs (static_assert sizes match)
- transport.h: abstract class (put/signal/wait/sync)
- executor.h: Executor::execute(PlanView, Config) -- for-loop over OpEntry
- mock.cpp: in-process memcpy transport
- Custom op stubs: `#ifdef ASCEND_ENABLED` gating
- torch_binding.cpp: `TORCH_LIBRARY(planck, m)` registration

Chunk 7 (torchair + Benchmarks): 2 tasks
- graph_pass.py: pattern match -> replace with Planck custom ops
- ops.py: FakeTensor registration for torch.compile tracing
- bench_vs_hccl.py: 3 groups (train 256/64MB, infer 16/64KB, KV pipeline)

---

## Next Step

Phase A code complete, all tests pass. 此文件的当前用途是记录提炼结果，供后续Phase B迭代参考。

Phase B进入条件: Ascend NPU hardware + CANN SDK。
macOS可预做: C++ headers + mock transport + PlanView (Chunk 6 partial)。
