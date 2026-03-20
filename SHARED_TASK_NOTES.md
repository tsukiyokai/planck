# Planck v0.1 -- Distilled Notes

## Status

Phase A code complete. 29/29 Rust tests, 4/4 Python tests, 0 warnings.
Benchmark: compile ~1.36us (红线 <1ms), instantiate ~73ns (红线 <1us).

---

## Design Document Distillation (2026-03-20)

> 精读 `docs/plans/2026-03-19-planck-design.md` 后的核心要素提炼。

### 1. 三层竞争优势 (相乘关系)

```
L3: Pattern Specialization  -- PanGu通信pattern先验 -> skip/prefetch/partial-reduce
L2: Plan Compilation         -- AOT全图编译 -> cross-op buffer reuse / schedule reorder
L1: Hardware Exploitation    -- MTE+AIV+Cube物理隔离 -> 通信/压缩/计算零竞争
```

竞品对比:
- L1: GPU的Comet/FLUX/DeepEP争抢同一SM池; Ascend引擎物理隔离无竞争
- L2: NCCL/HCCL逐op决策无跨op视图; Planck编译整个CommGraph
- L3: 通用库只见孤立API调用; Planck知道PanGu全部通信模式

### 2. 系统架构: Compile-Execute分离 + 双渠道交付

```
Rust Plan Compiler (brain)     C++ Custom Ops + Executor (hands)
  topo/cost/algo/sched           ACL Runtime/kernel/HCCL P2P
           |                                |
           +---- PlanCache (PyO3 FFI) ------+
           |                                |
    Channel A: Standalone          Channel B: ACL Graph
    (core, framework-agnostic)     (torchair, zero graph break)
```

关键决策:
- Standalone是核心不是fallback (NCCL路径: 先独立库后CUDA Graph)
- 两渠道共享Compiler/Transport/CustomOps,只有交付方式不同
- Rust/C++分界: Rust决策(编译期), C++执行(运行期), FFI仅PlanCache
- 构建: maturin -> cargo(Rust) + corrosion(cmake/C++) -> unified wheel

### 3. Plan IR三层结构

```
CommGraph (L1)        LogicalPlan (L2)           ExecutionPlan (L3)
"做什么"               "怎么分解"                  "怎么执行"
(语义层,硬件无关)      (算法已选,buffer已规划)     (stream分配,零运行时决策)
```

序列化: C packed structs (header 32B + BufEntry[] 12B each + OpEntry[] 16B each)
选packed struct而非FlatBuffers: plan结构简单,reinterpret_cast直读,无需IDL

### 4. 9条原语指令 (MSCCL++ one-sided model)

基础(5): Put / Signal / Wait / LocalCopy / LocalReduce
融合(3): PutWithSignal / WaitReduceCopy / WaitReducePut (编译器Pass 5自动生成)
同步(1): Noop

选one-sided(put/signal/wait)而非two-sided(send/recv): 解锁all-pairs算法,减少同步
MSCCL++报告小消息AllReduce 3.1x加速

### 5. 6个编译优化Pass

```
Pass 1: Algorithm Selection    AllReduce -> Ring(RS+AG)              决策层
Pass 2: Chunking & Pipeline    256MB/4chunks = 64MB, 4-stage        决策层
Pass 3: Buffer Planning        静态lifetime, cross-op buffer复用    资源层
Pass 4: Dependency Refinement  op-level -> chunk-level deps         资源层
Pass 5: Fusion                 Wait+Reduce+Put -> WaitReducePut     执行层
Pass 6: Inline Transform       插入Quantize/Dequantize, MTE+AIV    执行层(Ascend专属)
```

Pass 6特殊: 利用MTE+AIV物理隔离,pipeline填满后transform延迟完全隐藏

### 6. v0.1 Scope

包含: 8卡HCCS | AllReduce Ring | buffer+schedule opt | plan template |
      torchair pass | pipelined/quantized AllReduce | KV pipeline | 3组benchmark
不含: MC2 | 多机RoCE | 通用拓扑发现 | 多算法选择 | MoE

成功标准:
- 2x AllReduce pipeline < HCCL 2x独立AllReduce
- quantized AllReduce: 同busBW, 50%数据量
- KV pipeline: first-token-to-decode < naive sequential
- 所有custom ops可被ACL Graph capture

### 补充: 关键Trade-off

- KV cache pipeline用standalone executor: AllToAll的D2H会break ACL Graph capture
- v0.1不做通用拓扑/多算法: specialize-first,先让PanGu最快
- Rust性能价值不在通信热路径,在3个杠杆: 更深搜索/us级实例化/ms级重编译

---

## Implementation Distillation

> 精读 `docs/plans/2026-03-19-planck-v01-implementation.md` 后的结构化提炼。
> 聚焦 Chunks 1-5 (Phase A, Rust + Python)。

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

## Lockstep仿真逻辑深度分析

> 精读 plan.rs:476-681 两个仿真测试。
> 目标: 为C++ executor和planck-sim提供可复现的参考语义。

### 两层验证体系

| 测试                        | 验证层级            | 数据结构                     | 核心验证目标               |
|:----------------------------|:--------------------|:----------------------------|:--------------------------|
| `simulate_ring_allreduce`   | 算法层 (algo.rs)    | `data[rank][chunk][elem]` 3D | chunk index公式正确性      |
| `simulate_plan_execution`   | 全编译管道          | `data[rank][flat]` 1D + BufEntry | sched+fuse+execute端到端  |

两测试用相同参数: 8 rank, msg_size=256B, 64个f32, 期望结果=[36.0; 64] (1+2+...+8=36)。

### simulate_ring_allreduce (plan.rs:476-537) — 算法层

执行模型: 逐step lockstep，每step先snapshot所有rank的发送数据，再处理接收。

```
for step in 0..14:                    // 7 RS + 7 AG
    sends[r] = data[r][send_chunk]    // snapshot BEFORE mutate
    for r in 0..8:
        if RS: data[r][recv_chunk] += sends[src_rank]   // accumulate
        if AG: data[r][recv_chunk]  = sends[src_rank]   // overwrite
```

为什么先snapshot? 不snapshot的话，如果rank 0先处理完(修改了自己的data)，rank 1再发送时读到的就是rank 0的已修改数据。snapshot保证所有rank在同一step看到的是"上一step结束时"的一致状态。

### simulate_plan_execution (plan.rs:543-681) — 全管道

#### 内存模型

```
data[rank]:    Vec<f32>   大小nf=64    对应 BufPool::Input 和 BufPool::Output
scratch[rank]: Vec<f32>   大小由buffer table推算   对应 BufPool::Scratch
```

Input和Output共享同一个data数组(in-place AllReduce)。BufEntry的offset字段做byte->f32换算: `off = buf.offset / 4`。

scratch大小计算: 取所有Scratch BufEntry中`(offset + size)`的最大值，向上对齐到f32边界。

#### 执行模型: 双阶段逐指令

```
for op_idx in 0..num_ops:
    // Phase 1: 所有rank执行local ops + 暂存puts
    puts = []
    for rank in 0..8:
        op = plans[rank].ops[op_idx]
        match op.opcode:
            Noop | Signal | Wait  -> skip (lockstep隐式同步)
            Put | PutWithSignal   -> puts.push(dst_rank, dst_off, read(src_buf))
            LocalReduce           -> data[rank][dst] += scratch[rank][src]
            LocalCopy             -> data[rank][dst]  = scratch[rank][src]
            WaitReducePut         -> reduce then stage put (see below)
            WaitReduceCopy        -> reduce then copy (see below)

    // Phase 2: 将暂存的puts写入远端scratch
    for (dst_rank, dst_off, vals) in puts:
        scratch[dst_rank][dst_off..] = vals
```

#### Fused Op语义 (关键: _pad字段复用)

WaitReducePut (RS phase产生):
1. Reduce: `data[rank][dst_buf.off] += scratch[rank][src_buf.off]`
2. Put: 读reduce后的`data[rank][dst_buf.off]`，暂存到`puts`
3. _pad存储: put在远端的目标buffer index (即远端scratch的BufEntry index)

WaitReduceCopy (AG phase最后一步, RS最后结果已在input中):
1. Reduce: `data[rank][_pad.off] += scratch[rank][src_buf.off]`
2. Copy: `data[rank][dst_buf.off] = data[rank][_pad.off]`
3. _pad存储: reduce的目标buffer index (input chunk, 非output)
4. dst_buf: copy的目标 (output chunk)

#### 为什么双阶段(Two-Phase)是必须的

考虑: rank A执行Put写入rank B的scratch，同时rank B执行LocalReduce读自己的scratch。如果Put立即生效，rank B可能读到half-written数据。

双阶段保证: Phase 1中所有Put只暂存不生效，Phase 2统一apply。等价于"所有rank先完成本地计算，再通信"。

在真实硬件上: put+signal+wait协议天然提供这个保证——Wait阻塞直到Signal到达，Signal只在Put完成后发出。所以仿真跳过Signal/Wait是安全的。

#### 对称性不变量

`plans[0].ops.len() == plans[r].ops.len()` for all r。这由sched.rs保证: 每个rank的AlgoStep数相同(2*(n-1))，每步生成4个op，总数一致。fusion后仍对称(因为fusion pattern只依赖opcode序列，各rank的opcode序列结构相同)。

### C++ Executor复现清单

1. 9种opcode处理: 5基础 + 3 fused + Noop (fused必须正确读_pad)
2. BufPool路由: Input/Output -> 用户tensor, Scratch -> executor管理的buffer
3. 序号一致性: C++ for-loop顺序必须与Rust ops Vec顺序一致
4. 双阶段语义: 真实transport的put+signal+wait天然保证；mock transport需显式双阶段
5. in-place语义: Input和Output指向同一tensor (AllReduce场景)

### planck-sim复现清单

与C++ executor相同，但用纯软件模拟transport:
1. 可以直接复制Rust测试的双阶段逻辑
2. 需要支持多pipeline chunk (Rust测试只用chunks=1)
3. 验证: 跑相同参数，结果必须bit-exact

---

## Phase B macOS Design Distillation (2026-03-20)

> 精读 `docs/plans/2026-03-20-phase-b-macos-design.md` 和 `docs/plans/2026-03-20-phase-b-macos-impl.md` 后提炼。
> 聚焦5个关键技术点: _pad字段、MockWorld远端寻址、3轮notify、GET模式、InlineReduce重叠。

---

### 总览: 两个独立Block

```
Block 1: C++ Executor + PyTorch Eager    verify "data correct?"  (functional correctness)
Block 2: planck-sim DES                  verify "timeline good?" (schedule quality)
Independent, can be developed in parallel. Share plan bytes, not code.
```

不做: graph_pass.py / torchair集成 / 入图测试 / bench_vs_hccl.py (需Ascend硬件)。

---

### Block 1 关键设计决策

#### 决策1: _pad字段的双重语义 (核心FFI契约)

OpEntry._pad (u16) 在未融合指令中未使用,融合后承载不同语义:

| 融合指令          | _pad含义                    | Rust赋值 (plan.rs:266)         | C++读取 (engine.cpp)                 |
|:-----------------|:---------------------------|:------------------------------|:------------------------------------|
| WaitReducePut    | put的远端buffer index       | `ops[i+2].dst_buf` (Put的dst) | `transport.put(dst_rank,dst,size,op._pad,0)` |
| WaitReduceCopy   | reduce的目标buffer index    | `ops[i+1].dst_buf` (Reduce的dst) | `resolve(op._pad)` 作reduce目标+copy源 |
| PutWithSignal    | 未使用, 置0                 | `_pad: 0`                     | 不读取                               |

WaitReducePut执行顺序 (engine.cpp):
1. Wait: `transport.wait(op.dst_rank)` -- 等远端数据到达本地scratch
2. Reduce: `dst_buf[j] += src_buf[j]` -- 本地reduce (src=scratch, dst=input[chunk])
3. Put: `transport.put(op.dst_rank, dst, size, op._pad, 0)` -- _pad寻址远端

为什么_pad而非新增字段: OpEntry是16B (4字节flags+6个u16=16), repr(C)与C++的`#pragma pack(1)`镜像。
_pad是对齐产生的"免费"语义槽位,避免增大struct破坏ABI。

WaitReduceCopy执行顺序:
1. Wait: `transport.wait(op.dst_rank)`
2. Reduce: `buf[_pad] += buf[src_buf]` -- _pad是reduce目标 (input[chunk])
3. Copy: `buf[dst_buf] = buf[_pad]` -- 从reduce结果copy到output

#### 决策2: MockWorld远端buffer寻址 (buffer index抽象)

```
Transport::put(dst_rank, local_src, size, remote_buf_idx, offset)
                                         ^^^^^^^^^^^^^^^^
                                         not a raw pointer, but a buffer index
```

设计要点:
- put()第4参数是远端rank的buffer table index,不是本地指针
- MockWorld持有所有rank的buffer指针表: `ranks[dst_rank].resolve_buf(remote_buf_idx)`
- resolve流程: buf_idx -> BufEntry{pool, offset, size} -> pool决定base(input/output/scratch) -> base + offset

为什么用index而非指针:
- mock场景: 共享内存中所有rank的buffer都在同一进程,用index查找世界状态
- HCCS真机: remote_buf_idx映射到RDMA remote memory region key + offset
- 统一抽象: Transport接口不暴露地址空间差异

MockWorld同步: signals[src][dst] int矩阵 + 单mutex/condvar。
setup_rank(): 注册buffer地址 + plan的BufEntry表,供resolve_remote()使用。

#### 决策3: 构建隔离

C++ cmake和Rust cargo完全独立,通过plan bytes文件交互:
- gen_fixtures.py: 用PyO3调Rust编译器 -> 生成.bin文件
- C++ test: 读取.bin -> PlanView(zero-copy reinterpret_cast) -> Executor::execute()
- find_package(Torch QUIET): 有libtorch编译torch_binding,没有跳过
- 测试框架: 纯assert + test_util.h (~30行), 不引入Catch2/GTest

#### 决策4: PyTorch Eager路径

```
torch.ops.planck.pipelined_allreduce(a, b, plan_key)
  -> C++ TORCH_LIBRARY -> PlanCache -> Executor::execute(plan, config)
  -> MockTransport -> result tensor
```

FakeTensor注册: `@torch.library.register_fake("planck::allreduce")` -- torch.compile tracing用。

---

### Block 2 关键设计决策

#### 决策5: 3轮Notify握手 (AscendModel)

来源: HCOMM phase2-hcomm-platform源码。

```
AscendModel::notify_time(link) = notify_rounds * link.latency_us = 3 * 1.5us = 4.5us
```

物理含义: HCCS通信前的3轮信令握手:
1. 发送方通知接收方"我要写数据"
2. 接收方确认buffer就绪
3. 发送方确认开始传输

每次Wait操作都会计入此开销,是小消息通信的显著瓶颈。

#### 决策6: GET模式 vs PUT模式 (HCCS特有)

```
PUT: latency = 1 * lat + data_time     (sender pushes)
GET: latency = 2 * lat + data_time     (receiver requests then sender sends)
```

AscendModel用GET建模 (来源: prim_rules.cc):
`put_time = 2.0 * link.latency_us + size / (bw * 1e3)`

额外1个latency: GET先发请求包,等数据返回。vs NVLink PUT模式只需1个latency。

#### 决策7: InlineReduce重叠 (MTE+AIV物理隔离)

```
SimpleModel:  inline_reduce_put = reduce_time + put_time        (sum, no overlap)
AscendModel:  inline_reduce_put = notify_time + max(reduce, put) (overlap!)
```

物理根据: DaVinci架构的MTE(Memory Transfer Engine)和AIV(AI Vector)是独立流水线:
- MTE: DMA搬运 (put操作)
- AIV: 向量计算 (reduce操作)
- 两者可同时工作,取max而非sum。来源: dispatcher_pub.h (HCOMM)

验证: timing.rs测试 `assert!(fused < separate)` 确认重叠效果。

#### 决策8: DES引擎架构

```
Simulator { queue: BinaryHeap, clock: f64, links: Vec<LinkState>,
            signals: Vec<Vec<i32>>, waiting: Vec<Option<u16>> }
```

EventKind: OpStart / OpEnd / PutEnd(释放链路) / Unblock(signal到达唤醒wait)
链路竞争: `effective_bw = link_bw / active_flows` (公平共享)

Wait处理: signal到则立即消费; signal未到则设waiting[rank], 等Signal事件触发Unblock。

#### 决策9: Chrome Trace 4层嵌套

```
Collective (AllReduce 256MB)          pid=rank, tid=0
  Pipeline_Chunk (chunk 0/1/2/3)      B/E nesting
    Op (Put/Wait/Reduce/...)          X event
      HwAction (notify/dma)           X event, cat="hw"
```

+ flow event (rank间Put箭头) + counter event (链路利用率)

#### 决策10: TimingModel可插拔

```rust
trait TimingModel: put_time / notify_time / reduce_time / inline_reduce_put_time / kernel_launch_overhead
```

3种实现: SimpleModel(alpha-beta) / AscendModel(硬件感知) / CalibratedModel(真机校准, post-v0.1)。
sim模块用feature flag隔离: `--features sim`启用, 依赖toml+serde (可选)。

---

### 跨Block设计发现

1. _pad是Rust-C++ FFI隐式契约: C++ executor对_pad的解读必须与Rust fuse()的赋值完全一致。设计文档分散描述,需交叉阅读impl的engine.cpp和design的WaitReducePut说明。

2. Mock和真机Transport接口完全一致: `put(dst_rank, src, size, remote_buf_idx, offset)`。差异仅在实现层: MockWorld用memcpy, HCCS用RDMA+DMA engine。

3. DES仿真器不模拟数据: 只模拟时间。Block 1验证"数据对不对", Block 2验证"schedule好不好"。

4. AscendModel的6个参数全部来自HCOMM源码 (非猜测值):
   - 3轮notify: phase2-hcomm-platform
   - GET模式2x latency: prim_rules.cc
   - SQE队列深度2048: stream_pub.h
   - 门铃批处理每10WQE: send_recv_executor
   - InlineReduce max: dispatcher_pub.h
   - CQ轮询10us: transport_roce.cc

5. TOML配置默认值 (30 GB/s, 1.5us lat, 3 notify rounds, 460 GB/s HBM) 与topo.rs一致。校准时改TOML不改代码。

---

### 执行计划 (Phase B macOS)

Block 1 (C++ Executor):
- [ ] gen_fixtures: Python脚本生成8个rank的plan .bin文件
- [ ] impl_cpp_ffi: plan.h + transport.h + executor.h + mock.cpp + CMakeLists.txt + test_util.h (~250行)
- [ ] gate_ffi: cmake build + ctest test_plan
- [ ] impl_executor: engine.cpp 9种opcode, 特别注意WaitReducePut的_pad (~300行)
- [ ] gate_exec: 8线程仿真, 输出=[36.0, ...] (sum 1..8)
- [ ] impl_torch_eager: torch_binding.cpp + ops.py (~200行)
- [ ] gate_eager: pytest test_torch_eager.py

Block 2 (planck-sim DES):
- [ ] impl_sim_engine: Cargo.toml feature gate + config.rs + engine.rs + link.rs (~500行)
- [ ] gate_sim_core: cargo test --features sim
- [ ] impl_sim_trace: timing.rs (Simple+Ascend) + trace.rs (Chrome Trace) (~300行)
- [ ] gate_sim: 集成测试 (pipeline_overlap + monotonic + InlineReduce)
- [ ] impl_sim_pyo3: planck-python绑定 + simulate() API (~50行)
- [ ] gate_sim_py: pytest test_sim.py

### 风险

1. PlanView zero-copy: C++用reinterpret_cast直读, 依赖struct layout完全一致, static_assert是唯一安全网
2. MockWorld单mutex: 8线程够用, 更多rank需改同步方案
3. DES精度: analytical model, 8%误差目标来自Echo论文, 尚未校准
4. libtorch可选: macOS上可能没有, torch_binding会被跳过
5. sim feature: planck-python直接启用sim, maturin develop后toml/serde会编译进wheel

## Next Step

Phase B设计文档精读完成。下一轮建议从Block 1 gen_fixtures开始 (依赖最少, 验证FFI契约)。
两Block可并行,但Block 1优先 (验证_pad契约是最高风险项)。

## impl_topo分支验证 (2026-03-20)

目标: 实现 topo.rs — 8卡HCCS拓扑。
结论: topo.rs在Phase A Chunk 2中已完整实现，无需额外工作。
验证: 29/29 Rust测试通过(含topo模块3个测试 + 其他模块使用topo的测试)。

## impl_cost验证 (2026-03-20)

目标: 实现 cost.rs -- alpha-beta-gamma代价模型。
结论: cost.rs在Phase A Chunk 2 (Task 4)中已完整实现，无需额外工作。
验证: 3/3 cost模块测试通过(cost_from_topology, ring_allreduce_cost_scales_with_size, ring_cost_formula)。
内容: CostModel{alpha,beta,gamma} + from_topology() + ring_allreduce() + 3测试，共75行。
代码位置: crates/planck-core/src/cost.rs
编译管道集成: plan.rs的compile()中调用CostModel::from_topology()。

## impl_algo验证 (2026-03-20)

目标: 实现 algo.rs — Ring AllReduce算法分解。
结论: algo.rs在Phase A Chunk 3 (Task 5)中已完整实现，无需额外工作。
验证: 4/4 algo模块测试通过 + 29/29全部Rust测试通过。
内容: Phase枚举(RS/AG) + AlgoStep结构体 + ring_allreduce() + 4测试，共120行。
代码位置: crates/planck-core/src/algo.rs
编译管道集成: plan.rs的compile()中调用algo::ring_allreduce()生成步骤，传入sched::schedule()。

## impl_sched验证 (2026-03-20)

目标: 实现 sched.rs — 双缓冲流水线调度器。
结论: sched.rs在Phase A Chunk 3 (Task 6)中已完整实现，无需额外工作。
验证: 4/4 sched模块测试通过 + 29/29全部Rust测试通过。二次验证(本轮): cargo test 29 passed。
内容: ScheduleResult结构体 + schedule() + 4测试，共208行。
代码位置: crates/planck-core/src/sched.rs
编译管道集成: plan.rs的compile()中调用sched::schedule()生成OpEntry指令，传入fuse()。
关键设计: pipeline chunk c -> stream c, 每chunk分配N个input + N个output + 2个scratch(双缓冲), ring_piece = msg_size/(chunks*ranks)。

## gate_fuse_compile验证 (2026-03-20)

目标: 在plan.rs中添加指令融合pass(fuse)和顶层compile()函数。
结论: fuse()和compile()在Phase A Chunk 3 (Task 7)中已完整实现，无需额外工作。
验证: 29/29 Rust测试通过，含目标要求的全部4个测试:
- fusion_put_signal: Put+Signal -> PutWithSignal (2→1)
- fusion_wait_reduce_put: Wait+LocalReduce+Put -> WaitReducePut (3→1)
- fusion_preserves_unfusable: Noop+Put保持不变 (2→2)
- compile_produces_valid_plan: 256MB/4chunk, magic正确, ops>0且<224

代码位置:
- fuse(): plan.rs:237-324 (贪心滑动窗口, 最长优先匹配)
- compile(): plan.rs:329-356 (topo→cost→algo→sched→fuse→ExecutionPlan)
- CompileRequest/Collective: plan.rs:215-227

## Chunk 4验证 (2026-03-20)

目标: template.rs + 端到端仿真测试 + Criterion benchmark。
结论: Chunk 4 (Tasks 8-10)在Phase A中已完整实现，无需额外工作。
验证: 29/29 Rust测试通过 + 4个Criterion benchmark通过。

Task 8 - E2E仿真:
- simulate_ring_allreduce: 算法层lockstep, 8 rank各持[r+1;64], 验证AllReduce结果=[36.0;64]
- simulate_plan_execution: Plan层lockstep, 含fused op模拟, two-phase put/apply, 验证=[36.0;nf]
- 代码位置: plan.rs:476-537(algo仿真), plan.rs:543-681(plan仿真)

Task 9 - PlanTemplate:
- from_plan(): 冻结ops, buffer sizes线性参数化(offset_scale, size_scale)
- instantiate(): O(num_buffers), 实测74ns (红线<1us, 余量13x)
- 3测试: template_creation / template_instantiation / instantiation_is_fast
- 代码位置: template.rs:1-124

Task 10 - Criterion Benchmark:
- compile_256mb_4chunk: ~1.36us (红线<1ms, 余量~700x)
- compile_16kb_1chunk: ~586ns
- compile_1mb_2chunk: ~870ns
- instantiate_16kb: ~74ns (红线<1us, 余量~13x)
- 代码位置: benches/compile_bench.rs:1-51

Phase A全部Chunks (1-5)已完成。整个项目进入Phase B。

## Chunk 5 PyO3绑定 -- 三次验证均通过 (2026-03-20)

目标: 实现Chunk 5 PyO3绑定。
结论: 代码已在先前迭代中完整实现，多轮验证确认无误。

最新验证 (第三轮):
- `cargo test -p planck-core` 29/29 passed
- `maturin develop` 构建成功 (CPython 3.13, macOS arm64)
- `pytest tests/test_plan_compile.py` 4/4 passed (test_import/test_compile_allreduce/test_plan_cache/test_template_instantiate)
- 注意: `cargo test --all` 对planck-python crate显示0 tests (正常 -- PyO3 crate无Rust tests, 只有Python tests)

Phase A Chunks 1-5全部完成，项目进入Phase B等待状态。

## Phase A总结报告验证 (2026-03-20)

目标: 撰写Phase A实现总结报告。
结论: 报告已存在于 `docs/phase-a-summary.md`，经三重验证数据一致。

验证结果:
- cargo test: 29/29 passed (与报告一致)
- pytest: 4/4 passed (与报告一致)
- 代码行数: 1,688 total (与报告精确一致)
- Benchmark (本次运行 vs 报告数据):
  - compile_256mb_4chunk: 1.39us vs 1.36us (噪声范围)
  - compile_16kb_1chunk:  613ns  vs 581ns  (噪声范围)
  - compile_1mb_2chunk:   886ns  vs 871ns  (噪声范围)
  - instantiate_16kb:     77ns   vs 73ns   (噪声范围)

报告覆盖的5个维度: 实现内容(Section 2) / 测试结果(Section 3) / Benchmark(Section 4) / 遇到的问题(Section 5) / Phase B就绪度(Section 7)。无需额外工作。

---

## Phase B Chunk 5: DES Engine Core (Tasks 11-12)

目标: 实现planck-sim DES仿真器核心, 通过 `cargo test -p planck-core --features sim -- sim`

### 依赖分析

用户指定的6个文件(Cargo.toml, lib.rs, mod.rs, config.rs, engine.rs, link.rs)编译时依赖timing.rs和trace.rs(impl plan的Chunk 6 Task 13-14)。engine.rs imports `timing::TimingModel`, `trace::Trace`; mod.rs imports `timing::create_model`。

决策: 同步实现timing.rs和trace.rs完整版。代码量小(~250行),且属于同一sim/模块,拆分到Chunk 6只是计划粒度问题。

### 发现: impl plan WaitReducePut signal/wait bug

impl plan的engine.rs对WaitReducePut用 `op.dst_rank` 作Wait source:
```rust
let src = op.dst_rank as usize; // Bug: 这是Put目标(next), 不是Wait来源(prev)
```

根因: fuse()将WaitReducePut的dst_rank设为Put的dst_rank(ops[i+2]), 覆盖了Wait source(ops[i].dst_rank)。
- WaitReduceCopy不受影响: dst_rank = ops[i].dst_rank = Wait source
- WaitReducePut受影响: dst_rank = ops[i+2].dst_rank = Put destination

Ring 8卡验证: rank 0的WaitReducePut的dst_rank=1(next), 但Wait应等rank 7(prev)的signal。engine.rs会检查signals[1][0]=0, 永远为false → 死锁。

修复: fuse() WaitReducePut中 `wait_event: ops[i].dst_rank` 保存Wait source rank。engine.rs读 `op.wait_event` 获取。现有29个测试无任何检查wait_event值,修改安全。

### plan.rs额外修改

1. `#[derive(Debug, Clone)]` on ExecutionPlan — engine.rs需要plans.to_vec()
2. `impl TryFrom<u8> for Opcode` — 替代impl plan中的unsafe transmute

### 执行步骤

- [x] 理解需求 + 精读impl plan + 识别bug
- [x] 修改plan.rs: derive Clone + fuse bug fix + Opcode::from_u8()
- [x] 修改Cargo.toml: [features] sim = ["toml","serde"]
- [x] 修改lib.rs: #[cfg(feature="sim")] pub mod sim
- [x] 创建sim/config.rs: SimConfig + TOML解析 + parse_size_str
- [x] 创建sim/link.rs: LinkState(bw/active_flows公平共享)
- [x] 创建sim/trace.rs: Chrome Trace JSON + total_time
- [x] 创建sim/timing.rs: TimingModel trait + SimpleModel + AscendModel
- [x] 创建sim/engine.rs: DES核心(BinaryHeap + signal/wait + link竞争)
- [x] 创建sim/mod.rs: simulate() API + 4集成测试
- [x] cargo build --features sim — 0 errors, 0 warnings
- [x] cargo test --features sim -- sim — 17/17 passed
- [x] cargo test -p planck-core — 29/29 passed (无回归)
- [x] cargo test -p planck-core --features sim — 44/44 passed (29原有 + 15新增)

### 风险 (已解决)

1. signal/wait语义: 已修复,用wait_event携带Wait source → sim_completes_without_deadlock测试验证
2. ExecutionPlan Clone: derive(Debug, Clone)无副作用
3. Opcode from_u8: match-based实现替代unsafe transmute

### 产物清单

修改:
- crates/planck-core/Cargo.toml — +sim feature, +toml/serde可选依赖
- crates/planck-core/src/lib.rs — +cfg-gated sim module
- crates/planck-core/src/plan.rs — +derive Clone, +from_u8(), fix fuse() wait_event

新建:
- crates/planck-core/src/sim/mod.rs — simulate() API + 4集成测试
- crates/planck-core/src/sim/config.rs — SimConfig + TOML解析 + 3测试
- crates/planck-core/src/sim/engine.rs — DES核心 + 1测试
- crates/planck-core/src/sim/link.rs — LinkState + 2测试
- crates/planck-core/src/sim/timing.rs — TimingModel + Simple + Ascend + 3测试
- crates/planck-core/src/sim/trace.rs — Chrome Trace JSON + 2测试

## Next Step

Chunk 5 (Tasks 11-12) + Chunk 6 (Tasks 13-14)已完成。下一步:
- Chunk 7 Task 15: Rust集成测试(已在mod.rs中前置完成)
- Chunk 7 Task 16: PyO3 simulate()绑定 (crates/planck-python/src/lib.rs)
- Chunk 7 Task 17: Python测试 + planck-sim.toml示例
