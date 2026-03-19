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

以下知识来自vibe-opt skill，内联于此供-p模式使用。

### 根本矛盾

计算能力(FLOPS)扩展远快于通信带宽，通信是制约scaling的核心瓶颈。

核心trade-off:
- Latency vs Bandwidth: 小包受启动延迟主导，大包受带宽主导
- 算术强度(Arithmetic Intensity): 计算量/通信量比值决定瓶颈位置(Roofline Model)
- alpha-beta模型: T = alpha + n/BW。alpha主导(decode阶段AllReduce)还是BW主导(训练梯度同步)决定优化方向完全不同

### 集合通信算法

Ring AllReduce:
- 数学本质: 在拓扑图上找哈密顿回路
- 通信量与节点数p无关: 每个节点收发 2(p-1)/p * N
- 带宽最优，但大规模下启动latency影响大
- RS step k: send chunk (rank-k+n)%n, recv chunk (rank-k-1+n)%n
- AG step k: send chunk (rank-k+1+n)%n, recv chunk (rank-k+n)%n
- dst = (rank+1)%n, src = (rank-1+n)%n

Tree算法:
- Double Binary Tree: NCCL用两棵互补树优化带宽利用率
- Recursive Doubling(k-nomial): 小消息AllReduce最优解，复杂度log_k(N)

算法选择: Ring→大包+少节点, Tree→小包或大规模, NCCL实际逻辑: 小包→Tree, 大包→Ring, NVLink充足→NVLS

### 硬件互连

| 层级 | 技术 | 带宽量级 | 场景 |
|------|------|---------|------|
| 卡间(Scale-up) | NVLink | 900 GB/s (NVL72) | 节点内GPU互连 |
| 卡间(Scale-up) | 华为灵衢(UB) | - | 昇腾超节点互连 |
| 卡间(Scale-up) | HCCS | ~30 GB/s per link | 昇腾8卡全互连 |
| 卡间 | PCIe Gen5 | 64 GB/s | 通用互连 |
| 节点间(Scale-out) | RDMA(IB/RoCE) | 400 Gbps+ | 跨机通信 |

### 性能建模

busBW修正(nccl-tests报告方式):
- busBW = data_size / time * 算法修正因子
- AllReduce: 2(p-1)/p, AllGather/ReduceScatter: (p-1)/p
- busBW接近硬件峰值 = 算法已最优

overlap效率: eta = 1 - max(T_comp, T_comm) / (T_comp + T_comm)
- eta=1完美重叠, eta=0完全串行
- 降低eta的真凶: SM争抢、同步barrier、pipeline bubble、HBM带宽争抢

尾延迟: 集合通信是木桶效应，T_collective = max(所有rank完成时间)
- 利用率逼近100%时排队延迟指数爆炸(Little's Law)，实际上限~80%

### Overlap技术

- MC2: 用AIV/MTE专用引擎搬数据不占算力SM
- FLUX: 通信融入GEMM tile头尾
- DeepEP: hook-based overlap不占SM资源
- 本质: 在SM争抢约束中找最优资源分配

## 性能分析维度框架

| 编号 | 维度 | 核心问题 |
|------|------|---------|
| 01 | 拓扑感知算法 | 通信算法是否充分利用硬件拓扑? |
| 02 | 计算-通信重叠 | 通信延迟能否被计算掩盖? |
| 03 | 传输层优化 | 底层数据搬运是否高效? |
| 04 | 内存与零拷贝 | 数据搬运路径是否存在冗余拷贝? |
| 05 | 通信压缩/低精度 | 能否通过减少通信数据量提升性能? |
| 06 | 调度与流水线 | 通信任务的调度是否最优? |
| 07 | 集合操作算法 | 具体集合操作的算法实现是否最优? |

症状→维度速查:
- 延迟高(小消息) → 03-transport, 06-scheduling
- 延迟高(大消息) → 07-collective-algo, 05-quantized-comm
- 延迟高(跨节点) → 01-topology-algo, 03-transport
- 带宽低(单操作) → 07-collective-algo, 04-memory
- 带宽低(端到端) → 02-overlap, 06-scheduling
- GPU利用率低 → 02-overlap, 06-scheduling

## 命名

Planck = Plan + Communication + Link
致敬Max Planck和物理学最小量子 -- 把通信优化做到最小粒度(chunk-level pipeline)，编译期确定一切。
