# Planck Design Deep-Read: Architecture Distillation

> 目标: 精读 docs/plans/2026-03-19-planck-design.md，提炼六大核心概念
> 状态: 第一轮 — 完成

---

## 1. 三层竞争优势 (Section 1.1)

三层之间是乘法关系，不是加法。每层独立成立，组合后效果倍增。

```
Layer 3: Pattern Specialization (模型特化)
  输入: PanGu的完整通信模式
  能力: partial-reduce / sparse / prefetch / skip冗余通信
  对手盲区: NCCL/HCCL只看到孤立的API调用，没有全局视图

Layer 2: Plan Compilation (架构优势)
  输入: 整张通信图(CommGraph)
  能力: 跨op buffer复用 / 调度重排 / 算法预选
  对手盲区: NCCL/HCCL是per-op模型，无法做跨op全局优化

Layer 1: Hardware Exploitation (硬件特化)
  输入: Ascend的MTE + AIV + Cube物理分离引擎
  能力: 通信/压缩/计算真并行，零资源争用
  对手盲区: GPU上Comet/FLUX/DeepEP都在抢同一个SM池
```

核心洞察: Layer 2是架构护城河。NCCL要获得这个能力需要从根本上重构其per-op执行模型，
这不是加个feature能解决的。Layer 1和Layer 3则分别是硬件绑定和场景绑定的护城河。

---

## 2. 系统架构 (Section 2)

### 2.1 compile-execute分离

这是整个架构的第一性原理: 编译期做所有决策，执行期机械执行。

```
Compile-time (Rust Plan Compiler):       Execute-time (C++ Custom Ops):
  topo analysis                             for-loop over op sequence
  algo selection                            zero decisions
  schedule optimization                     zero allocations
  buffer planning                           zero algorithm selection
  quantization decisions                    inline error check (1 cmp)
```

### 2.2 双渠道交付

两个渠道共享: Plan Compiler + Transport Layer + Custom Ops。差异仅在交付和执行方式。

```
Channel A (Direct Execution)             Channel B (ACL Graph)
  Standalone Executor                      torchair graph pass
  ProcessGroup后端                          custom ops inside ACL Graph
  Python/C API                             graph-compatible execution
  framework-agnostic                       seamless Ascend integration
  灵活性最大                                  推理零graph break
```

战略定位: Channel A是核心(不是fallback)，Channel B是优化渠道。
类比NCCL: 先独立库，后加CUDA Graph集成。

### 2.3 Rust/C++分工

```
Rust = "大脑" (Plan Compiler)
  petgraph拓扑图、算法选择、调度优化、buffer规划
  FFI出口: PlanCache -> PyO3 -> Python -> C++ custom ops读取参数

C++ = "双手" (Custom Ops + Executor)
  ACL Runtime调用、kernel launch、HCCL P2P调用、stream管理
```

FFI边界很窄: 仅PlanCache一个通道。这是刻意的——窄接口降低耦合。

---

## 3. Plan IR三层结构 (Section 3.1)

```
CommGraph (Layer 1 — 用户输入)
  语义: "对哪些tensor做哪些collective"
  特点: 算法无关、硬件无关
  类比: SQL查询(声明what，不管how)
          |
          v  [Pass 1: Algorithm Selection]
LogicalPlan (Layer 2 — 编译中间表示)
  语义: "这些collective分解为这些原语，带这些依赖"
  特点: 算法已选、buffer已规划、但未绑定硬件
  类比: SQL执行计划(已选index，但未绑到具体page)
          |
          v  [Pass 4-6: Dependency/Fusion/InlineTransform]
ExecutionPlan (Layer 3 — 最终输出)
  语义: "在这些stream上按这个顺序执行这些op"
  特点: 完全确定，运行时零决策
  类比: 编译后的机器码
```

设计洞察: 三层IR的关键价值是解耦。CommGraph变了(模型换了)只重新编译；
LogicalPlan不变意味着算法选择结果可缓存；ExecutionPlan是纯机械执行的。

---

## 4. 9条原语指令 (Section 3.2)

基于MSCCL++的单边模型(put/signal/wait)，不是NCCL的双边模型(send/recv)。

### 基础指令 (5条)

| Opcode      | 语义                     | 对应操作           |
|:------------|:------------------------|:-------------------|
| Put         | 异步单边写到远端rank      | RDMA write语义     |
| Signal      | 通知远端rank              | doorbell / flag    |
| Wait        | 阻塞等待远端signal        | poll / interrupt   |
| LocalCopy   | 设备内buffer拷贝          | memcpy_d2d         |
| LocalReduce | 设备内reduce(sum/max)     | element-wise op    |

### 融合指令 (3条, 编译器生成)

| Opcode          | 融合模式           | 生成条件                      |
|:----------------|:-------------------|:-----------------------------|
| PutWithSignal   | Put + Signal       | Put后紧跟Signal到同一rank     |
| WaitReduceCopy  | Wait + Reduce + Copy| Wait后立即Reduce再本地Copy    |
| WaitReducePut   | Wait + Reduce + Put | Wait后立即Reduce再Put到下一rank |

### 同步指令 (1条)

| Opcode | 语义         | 用途            |
|:-------|:-------------|:----------------|
| Noop   | 依赖同步点    | 无数据操作的barrier |

设计洞察: 融合指令不是用户写的，是Pass 5(Fusion)检测到Wait->Reduce->Put模式后自动生成。
单边模型的优势: MSCCL++实测小消息AllReduce 3.1x加速(消除不必要同步)。

---

## 5. 6个编译优化Pass (Section 3.3)

```
Pass 1: Algorithm Selection
  输入: CommGraph + Topology + CostModel
  决策: AllReduce 256MB, 8 ranks, HCCS -> Ring = ReduceScatter + AllGather
  依据: alpha-beta-gamma cost model + crossover公式

Pass 2: Chunking & Pipeline
  输入: LogicalPlan(单chunk)
  输出: LogicalPlan(多chunk, pipeline)
  例: 256MB / 4 chunks = 64MB each, 4-stage pipeline

Pass 3: Buffer Planning
  输入: 所有chunk的buffer需求
  输出: 静态生命周期分析 + 跨op复用
  例: chunk[0].RS.output复用为chunk[0].AG.input(同一块内存)

Pass 4: Dependency Refinement
  输入: op-level依赖图
  输出: chunk-level依赖图(更细粒度 -> 更多并行)
  核心: 把"AllReduce B依赖AllReduce A"细化为"chunk B[2]只依赖chunk A[2]"

Pass 5: Fusion
  输入: 相邻原语序列
  输出: 融合指令
  模式检测: Wait->Reduce->Put -> WaitReducePut

Pass 6: Inline Transform (Ascend特化)
  输入: Put/Wait指令
  输出: 插入AIV.Quantize(Put前) + AIV.Dequantize(Wait后)
  标记: MTE+AIV pipeline mode
  核心: 利用物理分离引擎隐藏压缩延迟
```

设计洞察: Pass 3(Buffer Planning)和Pass 4(Dependency Refinement)是Plan Compilation
相对per-op模型的核心优势所在。NCCL无法做跨op buffer复用，因为它看不到下一个op会用什么buffer。
Pass 6则是Ascend硬件特化的体现——GPU上没有物理分离引擎，做不了真正的零争用pipeline。

---

## 6. v0.1 Scope (Section 11)

### 交付范围

```
Plan Compiler [Rust]              Custom Ops [C++]
  8卡HCCS拓扑(硬编码)               pipelined_allreduce (2xAR流水线)
  AllReduce Ring(单算法)             quantized_allreduce (INT8压缩)
  Buffer规划 + 调度优化              KV cache pipeline transfer
  参数化模板(推理)                    (standalone executor)

torchair集成                       Benchmark (3组)
  识别AllReduce pattern              训练: 2xAR 256MB vs HCCL (带宽导向)
  替换为Planck custom op             推理: 2xAR 16KB vs HCCL (延迟导向)
  ACL Graph兼容性验证                 KV: pipeline vs naive (32层x64MB)
```

### 明确不做

MC2融合pass、多机RoCE、通用拓扑发现、多算法选择、MoE ops

### 成功标准

1. pipelined AllReduce: 2次AR总时间 < HCCL 2次独立AR
2. quantized AllReduce: busBW持平或更好，数据传输量减半
3. KV pipeline: first-token-to-decode延迟 < naive sequential
4. 所有custom ops可被ACL Graph capture(不break graph)

### 渐进路线

```
v0.1: 单机8卡 (HCCS)
v0.2: 小规模多机2-8节点 (HCCS + RoCE) + AIV-Direct探索
v0.3: 大规模64+机 (HCCS + RoCE + UB) + schedule合成
```

---

## 关键设计决策摘要

| 决策             | 选择                  | 理由                                     |
|:----------------|:---------------------|:----------------------------------------|
| 通信原语模型      | 单边(put/signal/wait) | MSCCL++ 3.1x加速证明; 解锁all-pairs算法     |
| Plan序列化格式    | C packed structs      | Plan结构简单; Rust/C++直接reinterpret_cast |
| 算法参数标定      | 自动microbenchmark     | 非硬编码阈值; 可适应不同硬件版本              |
| 调度合成         | 分层: 运行时查表+离线ILP | 运行时不能等待求解器; 部署前有充分时间优化      |
| Transport trait  | 单边put/signal/wait   | 与原语模型一致; 参考UCX capability probing   |

---

## 当前实现状态

Phase A (Rust + Python): 已完成
- topo.rs / cost.rs / algo.rs / sched.rs / plan.rs / template.rs
- sim模块(DES simulator)
- PyO3 bindings + Python PlanCache
- 测试: 7 pytest + 17 cargo test

Phase B: 部分完成
- Block 2 (sim) 已完成
- Block 1 (C++ execution layer) 需要Ascend硬件

## 下一步

本轮(第一轮)任务已完成: 设计文档精读 + 六大核心概念提炼。

---
---

# Implementation Plan Deep-Read: Phase A (Chunks 1-5)

> 目标: 精读 docs/plans/2026-03-19-planck-v01-implementation.md，提炼每个task的类型/测试/要求
> 状态: 第二轮 — 完成

## 全局依赖图

```
Chunk 1: Skeleton (Cargo workspace + maturin)
    |
    v
Chunk 2: IR Types + Topology + Cost Model
    |       plan.rs   topo.rs   cost.rs
    v
Chunk 3: Ring Algorithm + Scheduler + Fusion
    |       algo.rs   sched.rs   plan.rs(fuse/compile)
    v
Chunk 4: E2E Simulation + Templates + Benchmarks
    |       plan.rs(simulate)  template.rs  compile_bench.rs
    v
Chunk 5: PyO3 Bindings
    |       planck-python/lib.rs  python/planck/__init__.py
    v
Chunk 6: C++ Execution Layer  [needs Ascend NPU]
    |       csrc/include/  csrc/transport/  csrc/executor/
    v
Chunk 7: torchair + Benchmarks [needs Ascend NPU]
            python/planck/graph_pass.py  tests/bench_vs_hccl.py
```

## 文件结构 (Phase A实际产物)

```
crates/planck-core/src/
    lib.rs          re-exports (plan/topo/cost/algo/sched/template + sim[feature])
    plan.rs         IR types + serialize + fuse + compile + E2E sim tests
    topo.rs         8-card HCCS topology (56 directed links)
    cost.rs         alpha-beta-gamma cost model
    algo.rs         Ring AllReduce decomposition (AlgoStep)
    sched.rs        Pipeline scheduler (OpEntry/BufEntry generation)
    template.rs     Parameterized plan templates (inference JIT)
    sim/            [feature="sim"] DES simulator (config/engine/link/timing/trace)
crates/planck-core/benches/
    compile_bench.rs   criterion: compile <1ms, instantiate <1us
crates/planck-python/src/
    lib.rs          PyO3: PlanCompiler/PlanCache/PyPlanView/PyPlanTemplate/simulate
python/planck/
    __init__.py     re-export from _planck
tests/
    test_plan_compile.py   4 tests (import/compile/cache/template)
    test_sim.py            3 tests (json/config/pipeline)
```

---

## Chunk 1: Project Skeleton (Task 1)

已完成。Cargo workspace + maturin + rust-toolchain。

| 文件 | 要点 |
|:-----|:-----|
| Cargo.toml | workspace, resolver=2, release LTO=fat |
| rust-toolchain.toml | stable channel |
| pyproject.toml | maturin>=1.5, module-name=planck._planck |

---

## Chunk 2: Plan IR + Topology + Cost (Tasks 2-4)

### Task 2: Plan IR (plan.rs)

核心常量: `PLAN_MAGIC=0x4B4E_4C50` ("PLNK" LE), `PLAN_VERSION=1`

核心类型:

| 类型 | repr | 大小 | 字段 |
|:-----|:-----|:-----|:-----|
| PlanHeader | C | 32B | magic, version, num_ops/buffers/streams/events, num_ranks, my_rank, flags, _reserved[12] |
| BufEntry | C | 12B | pool(u32), offset(u32), size(u32) |
| OpEntry | C | 16B | opcode, stream_id, reduce_op, flags, src_buf, dst_buf, dst_rank, wait_event, signal_event, _pad |
| ExecutionPlan | -- | heap | header + Vec<BufEntry> + Vec<OpEntry> |

枚举:
- `Opcode(u8)`: Put=0..Noop=8 (5基础 + 3融合 + 1同步)
- `ReduceOp(u8)`: Sum=0, Max, Min
- `BufPool(u32)`: Scratch=0, Input, Output

序列化: header ++ buffers ++ ops字节拼接 (unsafe raw pointer, repr(C) zero-copy)

实现要点:
- `_pad`字段在融合指令中被重载: WaitReducePut存put目标buf, WaitReduceCopy存reduce目标buf
- `Opcode::from_u8()`显式枚举0-8校验

测试 (6项):
| 测试 | 断言 |
|:-----|:-----|
| header_is_32_bytes | sizeof == 32 |
| buf_entry_is_12_bytes | sizeof == 12 |
| op_entry_is_16_bytes | sizeof == 16 |
| opcode_values | Put=0, WaitReducePut=7, Noop=8 |
| header_magic | magic/version correct |
| serialize_roundtrip | serialize -> deserialize一致 |

### Task 3: Topology (topo.rs)

常量: `HCCS_BW_GBPS=30.0` (保守, 实际56 GB/s), `HCCS_LAT_US=1.5`

类型:
- `TransportType(u8)`: Hccs=0, Roce, Shm
- `Link`: src, dst, bandwidth_gbps, latency_us, transport
- `Topology`: num_ranks, links: Vec<Link>

方法:
- `hccs_8card()` -- 8卡全网格(all-to-all), 8*7=56条有向边
- `has_link(src, dst)` / `get_link(src, dst)` -- 线性查找
- `ring_order()` -- [0,1,...,7] (v0.1简单顺序)

测试 (3项): basics(56 links) / ring_neighbors / link_properties(Hccs, bw>0, lat>0)

### Task 4: Cost Model (cost.rs)

类型: `CostModel { alpha_us, beta_us_per_byte, gamma_us_per_byte }`

方法:
- `from_topology(topo)` -- alpha=latency, beta=1/(bw*1e3)
- `ring_allreduce(msg_size, n)` -- `T = 2(n-1)*alpha + 2(n-1)/n*M*beta + (n-1)/n*M*gamma`

测试 (3项): from_topology / scales_with_size(ratio>1000) / formula精确校验

---

## Chunk 3: Ring Algorithm + Scheduler + Fusion (Tasks 5-7)

### Task 5: Ring AllReduce (algo.rs)

类型:
- `Phase`: ReduceScatter, AllGather
- `AlgoStep`: phase, step, send_chunk, recv_chunk, dst_rank, src_rank, needs_reduce

函数: `ring_allreduce(num_ranks, my_rank) -> Vec<AlgoStep>`
- Ring: send to (r+1)%n, recv from (r-1+n)%n
- RS step k: send chunk (r-k+n)%n, recv chunk (r-k-1+n)%n, needs_reduce=true
- AG step k: send chunk (r-k+1+n)%n, recv chunk (r-k+n)%n, needs_reduce=false
- 总步数: 2*(n-1) = 14 for 8 ranks

测试 (4项): step_count(14) / send_recv_ranks / rs_chunk_indices / all_chunks_covered

### Task 6: Pipeline Scheduler (sched.rs)

类型: `ScheduleResult { ops, buffers, num_streams }`

函数: `schedule(steps, msg_size, pipeline_chunks) -> ScheduleResult`
- 每pipeline chunk c -> stream c
- 缓冲区: N input子块 + N output子块 + 2 scratch(双缓冲)
- RS step -> Put + Signal + Wait + LocalReduce (4 ops)
- AG step -> Put + Signal + Wait + LocalCopy (4 ops)
- 双缓冲: step%2选scratch_a或scratch_b

测试 (4项): produces_ops / uses_c_streams / buffer_sizes(>=2*chunks) / op_sequence(1chunk=56ops)

### Task 7: Fusion + Compile (plan.rs additions)

新类型:
- `Collective`: AllReduce
- `CompileRequest`: collective, msg_size, reduce_op, num_ranks, my_rank, pipeline_chunks

函数:
- `fuse(ops)` -- 贪心最长匹配:
  - 3-op: Wait+LocalReduce+Put -> WaitReducePut
  - 3-op: Wait+LocalReduce+LocalCopy -> WaitReduceCopy
  - 2-op: Put+Signal -> PutWithSignal
- `compile(req, topo)` -- topo -> cost -> algo -> sched -> fuse -> ExecutionPlan

测试 (4项): fusion_put_signal / fusion_wait_reduce_put / fusion_preserves_unfusable / compile_valid_plan

---

## Chunk 4: E2E Simulation + Templates + Benchmarks (Tasks 8-10)

### Task 8: E2E Simulation (plan.rs test-only)

两个仿真测试:
1. `simulate_ring_allreduce` -- 算法级: 8卡 256B 1chunk, data=[r+1;64], expect=[36.0;64]
2. `simulate_plan_execution` -- 指令级: compile() -> OpEntry lockstep执行 -> 验证正确性
   - 两阶段: Phase1收集本地ops+stash puts, Phase2 apply远端scratch
   - 融合ops展开为原始语义, 精度容差1e-4

### Task 9: Templates (template.rs)

类型:
- `ParamSlot`: MsgSize
- `BufExpr`: pool, offset_scale(f64), size_scale(f64)
- `PlanTemplate`: frozen_ops, buffer_exprs, header, param_slots, base_msg_size

方法:
- `from_plan(plan, base_msg_size)` -- 冻结ops, buffer sizes转scale系数
- `instantiate(msg_size)` -- O(num_buffers): scale * msg_size -> BufEntry

测试 (3项): creation / instantiation(同ops不同buffers) / is_fast(1000次<10ms)

### Task 10: Criterion Benchmarks (compile_bench.rs)

| Benchmark | 配置 | 目标 |
|:----------|:-----|:-----|
| compile/256MB_4chunk | 256MB, 4 pipeline | <1ms |
| compile/16KB_1chunk | 16KB, 1 pipeline | <1ms |
| compile/1MB_2chunk | 1MB, 2 pipeline | <1ms |
| instantiate_16KB | 16KB模板 | <1us |

---

## Chunk 5: PyO3 Bindings (Task 11)

### Rust侧 (planck-python/lib.rs)

| pyclass | 职责 | 关键方法 |
|:--------|:-----|:---------|
| PlanCompiler | AOT编译入口 | hccs_8card(), compile_allreduce(), compile_template() |
| PlanCache | 编译缓存 | hccs_8card(), get_allreduce(key=(msg_size,my_rank)), cache_size(), clear() |
| PyPlanView | plan只读视图 | num_ranks/ops/buffers/streams(), to_bytes() |
| PyPlanTemplate | 参数化模板 | instantiate(msg_size), num_ops() |

全局函数: `simulate(plans, config_toml=None) -> String(JSON)`

设计要点:
- GIL释放: compile/simulate用`py.allow_threads()`
- 缓存: Mutex<HashMap<(msg_size, my_rank), ExecutionPlan>>

### Python测试

test_plan_compile.py (4项): import / compile_allreduce / plan_cache / template_instantiate
test_sim.py (3项): json_output / config_toml / pipeline_more_events

---

## Chunks 6-7: Phase B (仅结构列表)

| 组件 | 文件 | 状态 |
|:-----|:-----|:-----|
| C++ plan.h / transport.h / executor.h | csrc/include/planck/ | 未创建 |
| Mock/HCCS Transport | csrc/transport/ | 未创建 |
| Custom Ops | csrc/ops/ | 未创建 |
| torchair graph pass | python/planck/graph_pass.py | 未创建 |
| HCCL benchmark | tests/bench_vs_hccl.py | 未创建 |
| DES Simulator | crates/planck-core/src/sim/ | 已完成 |

---

## 验证状态

```
cargo test -p planck-core              -> 29 passed
cargo test -p planck-core --features sim -> 44 passed
pytest tests/ -v                       -> 7 passed
```

## 关键设计决策 (实现层面)

| 决策 | 选择 | 理由 |
|:-----|:-----|:-----|
| 序列化 | repr(C) raw bytes | 不用serde, C++可reinterpret_cast直接读 |
| OpEntry._pad复用 | 融合指令重载_pad字段 | 避免扩展结构体, 保持16B对齐 |
| HCCS带宽 | 30 GB/s (实际56) | 保守估计, 标记calibratable |
| v0.1 events | wait/signal_event=0 | 独立pipeline chunks无需跨stream同步 |
| Template模型 | buffer线性缩放(f64) | buffer大小与msg_size成正比 |
| PlanCache key | (msg_size, my_rank) | pipeline_chunks和reduce_op作为编译参数但不入cache key |

## 计划与实现的差异发现

1. cost测试: impl plan写ratio>100, 实际代码assert ratio>1000 -- 更严格
2. test_plan_compile.py: impl plan写4个测试, 实际有4个(一致)
3. template.rs: impl plan定义BufExpr有base_offset/base_size/base_param, 实际实现简化为offset_scale/size_scale两个f64系数 -- 更简洁
4. sim模块: 不在原始impl plan的Chunks 1-7中, 是后续Phase B Block 2额外添加的
5. 测试总数: impl plan期望~29 core tests, 实际29(一致); sim额外15个

## 下一步

第二轮任务已完成: 实现计划精读 + 11个task的类型/测试/要求提炼。

---

# Chunk 1 骨架验证 (DAG Node)

> 状态: NODE已完成 — Chunk 1目标在先前Phase A实现中已超额完成

DAG node要求"Chunk 1项目骨架(只放pub声明和空函数签名)"，但实际代码已推进到Phase A全部完成:

- Cargo workspace + rust-toolchain.toml + pyproject.toml: 在Phase A第一步已建立
- planck-core 6个模块: 不仅有类型声明，还有完整实现 + 29 tests passing
- planck-python PyO3: 完整bindings + 7 pytest passing
- sim模块: Phase B Block 2额外完成
- `cargo build` 零warning, `cargo test` 29 passed

无需回退到骨架状态，当前实现是骨架的超集。

### 2026-03-21 再次验证

重新执行验证，确认状态未退化:
- `cargo build`: 零warning通过
- `cargo test`: 29 passed, 0 failed
- Chunk 1全部5项要求均满足(workspace/toolchain/core-modules/pyo3/maturin)
- 结论不变: NODE_COMPLETE

---

# DAG Node: impl_ir (Plan IR类型 + C-compatible序列化)

> 状态: NODE已完成 — 全部需求在Phase A实现中已覆盖

## 需求核对

| 需求 | 文件:行 | 验证 |
|:-----|:--------|:-----|
| PlanHeader 32B repr(C) | plan.rs:43-56 | test header_is_32_bytes |
| BufEntry 12B repr(C) | plan.rs:58-64 | test buf_entry_is_12_bytes |
| OpEntry 16B repr(C) | plan.rs:66-79 | test op_entry_is_16_bytes |
| Opcode (5+3+1) | plan.rs:11-23 | test opcode_values |
| ExecutionPlan serialize/deserialize | plan.rs:159-211 | test serialize_roundtrip |
| TDD magic验证 | plan.rs:402-406 | test header_magic |

额外已完成(超出本node要求): fusion pass, compile pipeline, E2E simulation tests

## 验证结果

- `cargo test -p planck-core`: 29 passed, 0 failed
- 零warning编译
- repr(C)大小断言全部通过
