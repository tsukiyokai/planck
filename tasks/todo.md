# Planck Task List

## Brainstorming Phase (2026-03-19)

- [x] 探索项目上下文和参考资料
- [x] 逐步澄清需求和约束
- [x] 提出架构方案并比较
- [x] 分节呈现设计
  - [x] Section 1: 项目定位与核心理念
  - [x] Section 2: 系统架构总览
  - [x] Section 3: Plan IR设计
  - [x] Section 4: Plan Executor
  - [x] Section 5: Custom Ops设计 (revised: ACL Graph协同)
- [x] 深度调研
  - [x] 集百家之长 Round 1 (15个项目)
  - [x] 集百家之长 Round 2 (24个项目)
  - [x] 高性能Rust编程
  - [x] NCCL内部架构
  - [x] 通信调度理论
  - [x] UCX/JAX/SHARP/RCCL
  - [x] ACL Graph协同模式
  - [x] AscendC自定义算子开发
- [x] 撰写设计文档
- [x] 调用writing-plans创建实施计划
  - Plan: `docs/plans/2026-03-19-planck-v01-implementation.md`

## Implementation Phase

See full plan: `docs/plans/2026-03-19-planck-v01-implementation.md`

### Phase A: Rust + Python (macOS, no hardware dependency)

- [x] Chunk 1: Project Skeleton (Cargo workspace + maturin)
- [x] Chunk 2: Plan IR Types + Topology + Cost Model
- [x] Chunk 3: Ring Algorithm + Pipeline Scheduler + Fusion
- [x] Chunk 4: Serialization + Templates + E2E Simulation Test
- [x] Chunk 5: PyO3 Bindings + Python PlanCache

### Phase B: C++ Execution Layer (requires Ascend NPU)

- [ ] Chunk 6: C++ Headers + Mock Transport + Executor + Custom Op Stubs
- [ ] Chunk 7: torchair Graph Pass + Benchmarks vs HCCL
