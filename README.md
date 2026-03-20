# Planck

Plan + Communication + Link — Ascend NPU上为PanGu大模型全栈特化的集合通信库。如同Planck常数定义了物理学的最小量子，Planck把通信优化做到最小粒度——chunk-level pipeline，编译期确定一切。

## 远景

通过AOT Plan Compilation实现跨操作全局优化，同时面向训练和推理。
战略定位: 独立通信库(类NCCL)，保留完全替代HCCL的可能性。

三层竞争壁垒(相乘关系):

```
L3  Pattern Specialization   PanGu通信pattern先验 -> skip/prefetch/partial-reduce
L2  Plan Compilation         AOT全图编译 -> cross-op buffer reuse / schedule reorder
L1  Hardware Exploitation    MTE+AIV+Cube物理隔离 -> 通信/压缩/计算零竞争
```

## 架构

```
Rust Plan Compiler (brain)          C++ Custom Ops + Executor (hands)
  topo / cost / algo / sched          ACL Runtime / kernel / HCCL P2P
            |                                    |
            +------- PlanCache (PyO3 FFI) -------+
            |                                    |
     Channel A: Standalone             Channel B: ACL Graph
     (framework-agnostic)              (torchair, zero graph break)
```

- Plan IR三层: CommGraph(做什么) -> LogicalPlan(怎么分解) -> ExecutionPlan(怎么执行)
- 9条单边原语: Put / Signal / Wait / LocalCopy / LocalReduce + 3融合指令 + Noop
- 6个编译Pass: 算法选择 -> 分块流水 -> Buffer规划 -> 依赖细化 -> 指令融合 -> 内联变换

## 技术栈

| 层         | 技术                                     | 用途                                |
|:-----------|:-----------------------------------------|:------------------------------------|
| Rust       | petgraph, PyO3                           | Plan Compiler (决策, 编译期)        |
| C++20      | AscendC, ACL Runtime                     | Custom Ops + Executor (执行, 运行期)|
| Python     | torchair, pytest                         | Graph pass + PyO3 bindings          |
| Build      | maturin + corrosion                      | cargo + cmake 统一 wheel            |

## 路线图

```
v0.1  单机8卡HCCS | AllReduce Ring | torchair集成 | 3组benchmark vs HCCL    <-- 当前
v0.2  多机RoCE | AIV-Direct | 更多算法 (Tree / Recursive Halving)
v0.3  大规模集群 | schedule合成 | MC2融合pass | MoE ops
```

## 当前进展

Phase A (Rust + Python, macOS, 无硬件依赖) — complete

| Chunk | 内容                                | 状态 |
|:------|:------------------------------------|:-----|
| 1     | Project Skeleton                    | done |
| 2     | Plan IR + Topology + Cost Model     | done |
| 3     | Ring Algorithm + Scheduler + Fusion | done |
| 4     | E2E Simulation + Templates + Bench  | done |
| 5     | PyO3 Bindings + PlanCache           | done |
| 6     | C++ Execution Layer                 | todo |
| 7     | torchair + Benchmarks vs HCCL       | todo |

测试: 29/29 Rust + 4/4 Python, 零warnings
性能: compile ~1.36us (红线 <1ms), instantiate ~73ns (红线 <1us)

## 代码统计

```
语言     行数    占比     说明
───────────────────────────────────────────────────────
Rust    1,615   95.7%    Plan Compiler + PyO3 bindings
Python     73    4.3%    Module exports + tests
C++         0    0.0%    Phase B 未启动
───────────────────────────────────────────────────────
Total   1,688  100.0%
```

核心文件:

```
plan.rs       682L   Plan IR + serialize + fuse + compile + E2E simulation
lib.rs(PyO3)  248L   PlanCompiler / PlanCache / PyPlanView / PyPlanTemplate
sched.rs      208L   Pipeline scheduler + double-buffered receive
template.rs   123L   Parameterized plan templates
algo.rs       119L   Ring AllReduce decomposition (RS + AG)
topo.rs       105L   8-card HCCS topology (56 directed links)
cost.rs        74L   Alpha-beta-gamma cost model
```

## 支持特性

- repr(C) Plan IR — 零拷贝 Rust-to-C++ FFI, reinterpret_cast 直读
- 8卡HCCS拓扑 — 8x7=56条有向链路, 可校准带宽/延迟
- Alpha-beta-gamma成本模型 — 从拓扑自动推导参数
- Ring AllReduce — ReduceScatter + AllGather, chunk索引数学验证
- Pipeline调度 — 4-chunk流水, double-buffered receive
- 指令融合 — PutWithSignal / WaitReduceCopy / WaitReducePut
- 参数化模板 — 冻结ops + 线性缩放buffer, ~73ns实例化
- 8-rank lockstep仿真 — 算法级 + plan级两套端到端验证
- PyO3绑定 — PlanCompiler / PlanCache, 编译期GIL释放
- Criterion基准 — 4种配置, 全部远超性能红线

## 成功标准

- 2x AllReduce pipeline < HCCL 2x独立AllReduce
- Quantized AllReduce: 同busBW, 50%数据量
- KV pipeline: first-token-to-decode < naive sequential
- 所有custom ops可被ACL Graph capture

## 创新源

39个项目/论文调研汇总，详见设计文档。核心参考:
MSCCL++ (one-sided primitives) / ACCL+ (FPGA overlay) / DeepEP (EP experts) /
FlashMoE (SSD offload) / ResCCL (residual compression) / CoCoNet (fusion DSL) /
TACCL (topology-aware synthesis) / TE-CCL (tensor-parallel) / Flux (GEMM-comm overlap)

## 文档

- 设计文档: `docs/plans/2026-03-19-planck-design.md`
- 实施计划: `docs/plans/2026-03-19-planck-v01-implementation.md`
- 精炼笔记: `SHARED_TASK_NOTES.md`
