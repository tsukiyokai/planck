# pgccl - PanGu Collective Communication Library

## 自动加载 Skill

在本项目目录下的对话中,当用户提出与通信性能、集合通信算法、HCCL/NCCL对比、硬件架构(HCCS/DaVinci/AI Core等)、带宽优化等相关问题时,自动调用 vibe-opt skill 进行分析和回答。

## 项目概要

面向Ascend NPU、为PanGu大模型全栈特化的集合通信库。通过AOT plan compilation实现跨操作全局优化,同时面向训练和推理。

核心架构: Rust Plan Compiler(决策) + C++ Custom Ops(执行) + ACL Graph协同(主路径)

设计文档: `docs/plans/2026-03-19-pgccl-design.md`

## 技术栈

- Rust: Plan Compiler(petgraph/rkyv/PyO3)
- C++20: Custom Ops + AscendC kernels + Standalone Executor
- Python: torchair graph optimization pass + PyO3 bindings
- 构建: maturin + corrosion(cargo+cmake集成)
