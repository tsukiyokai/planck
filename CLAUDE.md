# Planck - Plan + Communication + Link

## 项目概要

Planck是面向Ascend NPU、为PanGu大模型全栈特化的独立集合通信库。通过AOT plan compilation实现跨操作全局优化，同时面向训练和推理。保留完全替代HCCL的战略可能性。

核心架构: Rust Plan Compiler(决策) + C++ Custom Ops & Executor(执行) + 双渠道交付(直接执行/ACL Graph)

设计文档: `docs/plans/2026-03-19-planck-design.md`

## 技术栈

- Rust: Plan Compiler(petgraph/PyO3)
- C++20: Custom Ops + AscendC kernels + Standalone Executor
- Python: torchair graph optimization pass + PyO3 bindings
- 构建: maturin + corrosion(cargo+cmake集成)

## 通信优化领域知识

涉及通信性能、算法选择、代价模型、拓扑设计时，读取以下reference文件:

- ~/repo/_me/skills/vibe-opt/references/knowledge-base.md — 领域知识(硬件带宽、算法数学本质、性能建模)
- ~/repo/_me/skills/vibe-opt/references/perf-dimensions/00-dimension-index.md — 7维性能分析框架 + 症状→维度映射
- ~/repo/_me/skills/vibe-opt/references/papers.md — 论文索引(xCCL Survey、AllReduce优化等)

## 命名

Planck = Plan + Communication + Link
致敬Max Planck和物理学最小量子 -- 把通信优化做到最小粒度(chunk-level pipeline)，编译期确定一切。
