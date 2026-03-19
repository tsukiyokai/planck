# pgccl Design Document

> PanGu Collective Communication Library
> Date: 2026-03-19
> Status: Draft (brainstorming complete, pending implementation plan)

---

## 1. Project Vision

pgccl is a collective communication library for Ascend NPUs, specialized for PanGu large models. It targets both training and inference scenarios.

One-line positioning: Ahead-of-time plan compilation for cross-operation global optimization, exploiting Ascend's physically separated engines (MTE/AIV/Cube) for true communication/compression/computation parallelism.

### 1.1 Three-Layer Competitive Advantage

```
Layer 3: Pattern Specialization (PanGu-specific)
  Know the model's full communication pattern
  -> partial-reduce / sparse / prefetch / skip unnecessary communication
  -> Generic libraries see isolated API calls, no global view

Layer 2: Plan Compilation (architectural advantage)
  AOT compile the entire communication graph
  -> cross-op buffer reuse / schedule reordering / algorithm pre-selection
  -> NCCL/HCCL's per-op model cannot do cross-op global optimization

Layer 1: Hardware Exploitation (Ascend-specific)
  MTE + AIV + Cube physically separated engines
  -> communication/compression/computation truly parallel, zero resource contention
  -> On GPU, Comet/FLUX/DeepEP all compete for the same SM pool
```

The three layers multiply, not add.

### 1.2 Design Principles

| Principle         | Meaning                                                    |
|:------------------|:-----------------------------------------------------------|
| Minimalist        | 300 lines if possible, never 500                           |
| Specialize first  | Make PanGu fastest first, generalize later                 |
| Compile-time      | Zero algorithm selection, zero malloc, zero scheduling at runtime |
| Hardware-aware    | Exploit Ascend's physical separation, don't mimic GPU patterns |
| Verifiable        | Every milestone has a quantifiable performance comparison  |
| Learn from best   | 39 innovation sources, don't reinvent wheels               |

### 1.3 Dual Scenario Coverage

| Scenario  | Communication Profile           | pgccl Strategy                              |
|:----------|:--------------------------------|:--------------------------------------------|
| Training  | Large messages, fixed pattern   | Static plan, compile once run millions      |
| Inference | Small messages (decode), dynamic batch | Parameterized plan template, O(1) instantiation |

---

## 2. System Architecture

### 2.1 Architecture: Independent Library with Dual Delivery Channels

pgccl is an independent collective communication library, not a plugin for any specific framework.
Two equally important delivery channels serve different integration scenarios.

Key insight: HCCL communication ops CAN be captured inside ACL Graph. pgccl should exploit
this for seamless integration, while maintaining full standalone execution capability.

Strategic positioning: pgccl retains the possibility of fully replacing HCCL. The standalone
executor is the core (not a fallback), ACL Graph integration is an optimization channel.

```
                         pgccl Architecture

 +---------------------------------------------------------------+
 |                    pgccl Plan Compiler [Rust]                  |
 |  (core: topo analysis / algo select / schedule / buffer plan)  |
 |                                                                |
 |  Input: CommGraph                                              |
 |  Output: ExecutionPlan + optimization decisions                |
 +---------------+-----------------------------------------------+
 |               |                                                |
 |    Channel A (Direct)              Channel B (ACL Graph)       |
 |    (standalone, framework-agnostic) (torchair/graph mode)      |
 |    equally important               equally important           |
 |               |                                                |
 |  +------------v--------------+   +------------------------+   |
 |  | Standalone Executor [C++] |   | Graph Optimization Pass|   |
 |  | (full execution engine)   |   | (torchair integration) |   |
 |  |                           |   |                        |   |
 |  | binary plan -> execute    |   | pattern match -> replace|   |
 |  | (zero decision, fast/slow)|   | with pgccl custom ops    |   |
 |  +------------+--------------+   +------------+-----------+   |
 |               |                                |               |
 |  +------------v-------------------------------------------+   |
 |  | Transport Layer [C++]                                  |   |
 |  | HCCS provider | RoCE provider | UB provider | SHM      |   |
 |  +------------------------------------------------------------+
 |               |                                                |
 |  Channel A delivers:              Channel B delivers:          |
 |  - ProcessGroup backend           - torchair pattern replace   |
 |  - Python API (PyO3)              - custom ops in ACL Graph    |
 |  - C API (standalone)             - graph-compatible execution |
 |  - framework-agnostic             - seamless Ascend integration|
 +---------------------------------------------------------------+
```

### 2.2 Strategic Positioning: Independent Library

pgccl is designed as a self-contained communication library like NCCL, not a framework plugin.

NCCL's success path: independent library first, CUDA Graph integration added later as optimization.
pgccl follows the same path: standalone executor is the core, ACL Graph integration is an optimization channel.

Both channels share the same Plan Compiler, Transport Layer, and Custom Ops. The difference
is only in how the compiled plan is delivered and executed:
- Channel A: pgccl controls execution directly (full flexibility, works everywhere)
- Channel B: ACL Graph controls execution, pgccl provides optimized ops (zero graph break, best for inference)

This dual-channel design means pgccl can:
- Work with MindSpore, PyTorch, JAX, or any other framework
- Run on hardware where ACL Graph is not available (older Ascend versions)
- Accumulate enough depth to eventually fully replace HCCL
- Use ACL Graph integration as a "performance bonus", not a survival dependency

### 2.3 Compile-Execute Separation (Core Idea Unchanged)

```
Compile time (Rust Plan Compiler):
  - Topology analysis
  - Algorithm selection (Ring/Tree/RecursiveHD)
  - Schedule optimization (pipeline, overlap)
  - Buffer planning (lifetime analysis, reuse)
  - Quantization decisions

Execute time (Custom Ops in ACL Graph):
  - Mechanically execute pre-decided plan
  - HCCL P2P calls + AscendC kernels
  - Zero runtime decisions
```

### 2.4 Module Structure

```
pgccl/
+-- crates/                          # Rust workspace
|   +-- pgccl-core/                  # Plan Compiler
|   |   +-- src/topo.rs              # Topology graph (petgraph or CSR)
|   |   +-- src/cost.rs              # Alpha-beta-gamma cost model
|   |   +-- src/algo.rs              # Algorithm database
|   |   +-- src/sched.rs             # Schedule optimizer
|   |   +-- src/plan.rs              # Plan IR + serialization
|   |   +-- src/template.rs          # Parameterized templates
|   +-- pgccl-python/                # PyO3 bindings
|       +-- src/lib.rs               # PlanCache exposed to Python
|
+-- csrc/                            # C++ Custom Ops
|   +-- kernels/                     # AscendC kernels
|   |   +-- quantize_per_group.cpp
|   |   +-- dequantize_per_group.cpp
|   |   +-- reduce_add.cpp
|   +-- ops/                         # Op implementations
|   |   +-- pipelined_allreduce.cpp
|   |   +-- quantized_allreduce.cpp
|   |   +-- kv_pipeline_transfer.cpp
|   +-- executor/                    # Standalone executor (fallback)
|   |   +-- engine.cc
|   |   +-- stream_pool.cc
|   |   +-- mem_pool.cc
|   +-- transport/                   # Transport providers
|   |   +-- transport.h
|   |   +-- hccs.cc
|   |   +-- roce.cc
|   +-- compat/                      # torch compatibility
|   |   +-- process_group_pgccl.cc
|   +-- torch_binding.cpp            # TORCH_LIBRARY registration
|   +-- CMakeLists.txt
|
+-- python/pgccl/
|   +-- __init__.py
|   +-- plan_cache.py                # Rust PlanCache Python wrapper
|   +-- graph_pass.py                # torchair pattern replacement
|   +-- ops.py                       # FakeTensor registration
|
+-- tests/
    +-- test_quantize_kernel.py
    +-- test_custom_ops.py
    +-- test_graph_capture.py
    +-- bench_vs_hccl.py
```

### 2.5 Build System

```
maturin develop -> cargo builds Rust crates
                -> corrosion triggers cmake for C++
                -> linked as unified Python wheel
```

---

## 3. Plan IR Design

### 3.1 Three-Layer IR

```
Layer 1: CommGraph (user input)
  "What collectives to perform on which tensors"
  Semantic layer, algorithm/hardware agnostic

Layer 2: LogicalPlan (compiler intermediate)
  "These collectives decomposed into these primitives, with these dependencies"
  Algorithm selected, buffers planned, but not bound to hardware

Layer 3: ExecutionPlan (final output to C++ or Custom Ops)
  "Execute these ops in this order on these streams"
  Fully determined, zero decisions at runtime
```

### 3.2 Primitive Instruction Set (9 instructions)

Based on MSCCL++ one-sided model (put/signal/wait, NOT send/recv):

| Opcode           | Type     | Semantics                                    |
|:-----------------|:---------|:---------------------------------------------|
| Put              | Basic    | Async one-sided write to remote rank         |
| Signal           | Basic    | Notify remote rank                           |
| Wait             | Basic    | Block until signal from remote rank          |
| LocalCopy        | Basic    | Intra-device buffer copy                     |
| LocalReduce      | Basic    | Intra-device reduce (sum/max)                |
| PutWithSignal    | Fused    | Put + Signal (compiler-generated)            |
| WaitReduceCopy   | Fused    | Wait + Reduce + local copy                   |
| WaitReducePut    | Fused    | Wait + Reduce + Put to next rank             |
| Noop             | Sync     | Dependency sync point, no data operation     |

Fused instructions are NOT user-written; they are generated by the compiler's fusion pass when it detects patterns like Wait immediately followed by Reduce then Put.

### 3.3 Compiler Optimization Passes

```
Pass 1: Algorithm Selection
  AllReduce 256MB, 8 ranks, HCCS -> Ring: ReduceScatter + AllGather

Pass 2: Chunking & Pipeline
  256MB / 4 chunks = 64MB each, 4-stage pipeline

Pass 3: Buffer Planning
  Static lifetime analysis, cross-op buffer reuse
  Example: chunk[0].RS.output reused as chunk[0].AG.input (same memory)

Pass 4: Dependency Refinement
  Op-level deps -> chunk-level deps (enables more parallelism)

Pass 5: Fusion
  Detect Wait->Reduce->Put pattern -> WaitReducePut instruction

Pass 6: Inline Transform (Ascend-specific)
  Insert AIV.Quantize before Put, AIV.Dequantize after Wait
  Mark as MTE+AIV pipeline mode
```

### 3.4 Serialization Format

For the standalone executor path: C packed structs (not rkyv, not FlatBuffers).
Reason: Plan structure is simple (header + two fixed-length tables), both Rust and C++ can directly read via reinterpret_cast, no IDL needed.

```
Header (32 bytes): magic, version, num_ops, num_buffers, num_streams, num_events, flags
Buffer Table: BufEntry[] { pool, offset, size }
Op Table: OpEntry[] { opcode, stream_id, src_buf, dst_buf, dst_rank, wait_event, signal_event, flags }
```

For the ACL Graph path: optimization decisions are baked into custom op parameters, no separate serialization needed.

### 3.5 Parameterized Templates (Inference)

```
PlanTemplate = frozen graph structure + ParamSlot table
ParamSlot = { target: which op field, expr: arithmetic expression over runtime params }

Instantiation: evaluate ParamExpr for each slot, fill into op fields
  -> O(num_slots * ~10ns) = sub-microsecond for typical plans
  -> Enables per-request JIT plan specialization
```

---

## 4. Plan Executor (Standalone, Fallback Path)

### 4.1 Design: Layered Fast/Slow Path

```
Execution Context (per-invocation, mutable)
  buffer address binding, communicator handles, streams
  One Plan can create multiple Contexts (concurrent inference)

Fast Path (99.9% of execution)
  for-loop over pre-compiled op sequence
  zero decisions, zero allocations
  inline error code check (non-blocking, 1 cmp instruction)

Slow Path (exception only)
  Error classification (TensorRT-inspired):
    Level 0: Transient - network jitter, auto-retry
    Level 1: Degraded  - link bandwidth drop, switch to variant plan
    Level 2: Rank Failure - timeout, communicator shrink
    Level 3: Fatal - report to upper layer
  Plan variant switching (CUDA Graph multi-graph pattern)
  Communicator shrink (NCCL 2.27 ncclCommShrink pattern)
```

### 4.2 ACL Runtime Constraints

| Constraint            | Value                        | Impact on pgccl                  |
|:----------------------|:-----------------------------|:---------------------------------|
| Stream limit          | ~2048/device, 248 reserved   | Must pool, cannot create freely  |
| Huge page threshold   | >1MB uses huge pages         | Pre-allocate buffers >= 1MB      |
| Memory interaction    | Can be 30% of total time     | Eliminate D2H in plan            |
| HCCL stream reserve   | Part of reserved 248         | Coordinate with HCCL             |

### 4.3 Coexistence with ACL Graph

pgccl Plan Executor and ACL Graph are complementary, not competing:
- Compute kernels: ACL Graph (zero launch overhead)
- Communication: pgccl custom ops inside ACL Graph, or standalone executor for incompatible ops
- Coordination: stream events between ACL Graph nodes and pgccl ops

---

## 5. Custom Ops Design

### 5.1 Communication API Boundary

| Level        | NVIDIA              | Ascend              | pgccl v0.1? |
|:-------------|:--------------------|:---------------------|:------------|
| Kernel-level | NVSHMEM (public)    | AIV-Direct (not public) | No       |
| P2P          | ncclSend/Recv       | HcclSend/Recv       | Yes          |
| Collective   | ncclAllReduce...    | HcclAllReduce...    | Yes          |

pgccl v0.1 builds on HCCL P2P API. Kernel-level communication (AIV-Direct) is v0.2+ with Huawei collaboration.

### 5.2 Three Categories of Custom Ops

Category 1 - Scheduling Optimization (build better collectives from HCCL P2P):
- pgccl_pipelined_allreduce: cross-op pipeline of two AllReduces
- pgccl_kv_pipeline_transfer: layer-level pipeline KV cache transfer

Category 2 - Fusion (communication + computation merged):
- pgccl_rs_add_rmsnorm_ag: ReduceScatter -> Add+RMSNorm -> AllGather (MC2-style)
- pgccl_quantized_allreduce: quantized compressed AllReduce

Category 3 - Specialization (PanGu-specific patterns):
- pgccl_moe_dispatch: MoE expert dispatch (DeepEP-inspired)
- pgccl_partial_allreduce: partial synchrony TP (CAAT-Net-inspired)

### 5.3 Registration Pattern

```cpp
// TORCH_LIBRARY for schema definition
// TORCH_LIBRARY_IMPL for NPU implementation
// Python: @torch.library.register_fake for ACL Graph capture compatibility
```

### 5.4 Graph Optimization Pass

pgccl registers pattern replacements via torchair.register_replacement():
- Two adjacent AllReduces -> pgccl.pipelined_allreduce
- Large AllReduce -> pgccl.quantized_allreduce
- MatMul+AllReduce+Add+RMSNorm -> pgccl.rs_add_rmsnorm_ag

Plan Compiler decisions are passed to custom ops via PlanCache (Rust -> PyO3 -> Python).

---

## 6. Transport Abstraction

### 6.1 Trait Design (Rust)

```rust
trait Transport: Send + Sync {
    fn capabilities(&self) -> TransportCaps;
    fn put(&self, dst: Rank, src: &Buffer, dst_buf: &Buffer) -> Hook;
    fn signal(&self, dst: Rank);
    fn wait(&self, src: Rank);
    fn eager_threshold(&self) -> usize;
    fn alpha(&self) -> Duration;  // startup latency
    fn beta(&self) -> f64;        // per-byte transfer time
}
```

One-sided model (put/signal/wait), not two-sided (send/recv).
Inspired by MSCCL++, UCX capability probing, Gloo transport abstraction.

### 6.2 Providers

- HccsTransport: v0.1, intra-node via HCCS
- RoceTransport: v0.2, inter-node via RoCE
- UbTransport: v0.3, CloudMatrix UB mesh
- ShmTransport: same-die shared memory

### 6.3 Hierarchical Factorization (HiCCL-inspired)

```
{8}       -> single machine 8 cards (HCCS)
{8, 2}    -> 2 machines (HCCS + RoCE)
{8, 8, N} -> large scale (HCCS + RoCE + UB)
```

Each level uses its own transport backend. Algorithm code is shared across levels.

---

## 7. Theoretical Foundations

### 7.1 Cost Model

Hierarchical alpha-beta-gamma:
```
T_collective(algo, topo, msg_size) =
    sum over hierarchy level h:
        alpha_h * steps(algo, h)
      + beta_h  * volume(algo, h, msg_size)
      + gamma_h * compute(algo, h, msg_size)
```

Parameters calibrated via automated microbenchmark, not hardcoded thresholds.

### 7.2 Algorithm Selection

```
crossover = alpha * log(p) / beta
if msg_size < crossover: latency-optimal (recursive doubling, Bruck)
else: bandwidth-optimal (Rabenseifner, Ring, pairwise)
```

### 7.3 Schedule Synthesis (Layered)

- Runtime fast path: pre-computed cost table lookup (NCCL-style)
- Deployment offline: MCF/ILP-based synthesis (TE-CCL/SyCCL for large topologies)

---

## 8. Innovation Sources (39 Projects/Papers)

### 8.1 Plan IR & Algorithm
- MSCCL/MSCCL++: chunk-oriented DAG + put/signal/wait primitives
- HiCCL: M/R/Fence three-primitive composition + hierarchical factorization
- TACCL: sketch-guided algorithm synthesis
- SyCCL (SIGCOMM'25): symmetry-based large-scale schedule synthesis
- TE-CCL (SIGCOMM'24): MCF-based scheduling
- Swing: multi-port bandwidth-optimal AllReduce
- Rabenseifner: RS+AG decomposition of AllReduce

### 8.2 Hardware Exploitation
- ACCL+ (OSDI'24): inline compression on data path -> Ascend MTE+AIV mapping
- T3 (ASPLOS'24): compute-then-move hardware trigger
- NCCL NVLS: in-switch reduction
- SHARP: in-network aggregation
- MultiShot: NVSwitch multicast AllReduce
- NCCL 2.28 Device API: GPU-initiated networking (GIN)

### 8.3 Compute-Communication Fusion
- FlashMoE (NeurIPS'25): persistent kernel device-initiated, 9x GPU utilization
- Comet (MLSys'25): TB-level compute/comm isolation, 1.96x MoE layer
- FLUX: tile-level GEMM+comm fusion
- Domino: tensor slicing overlap
- DeepEP: hook-based SM-free overlap
- ResCCL (SIGCOMM'25): primitive-level TB scheduling, 77.8% less SM

### 8.4 Communication Reduction
- CAAT-Net (NeurIPS'25): partial synchrony TP, 50% less communication
- ZeCO (NeurIPS'25): zero-comm sequence parallelism, 60% speedup at 256 GPU
- Radius (MLSys'25 Outstanding): sparse AllReduce, 19% training speedup
- Flash Communication: INT4/INT8 quantized AllReduce
- Bagua: ByteGrad/QAdam/decentralized SGD
- PowerSGD: low-rank gradient compression
- PALSGD: pseudo-async local SGD

### 8.5 MoE Specialization
- Occult (ICML'25): collaboration-aware placement
- C2R (NAACL'25): routing-constrained reduction
- NetMoE (ICLR'25): dynamic sample placement
- PopFetcher (ATC'25): predictive expert prefetch
- MoE++ (ICLR'25 Oral): zero-computation experts
- ScMoE (ICML'25): shortcut-connected EP

### 8.6 Transport & Scheduling
- UCX: 3-layer UCS/UCT/UCP + capability probing
- Gloo: pluggable transport abstraction
- UCCL: multi-path + software CC
- NCCL: channel/proxy/tuner architecture (lessons from weaknesses)
- Crux (SIGCOMM'24): GPU-intensity priority scheduling

### 8.7 API & Programming Model
- JAX/XLA GSPMD: declare sharding, auto-insert communication
- OneFlow SBP: S/B/P declarative distribution
- RCCL: lessons (don't fork, need multi-algorithm support)

### 8.8 Rust Engineering
- petgraph: topology graph algorithms
- rkyv: zero-copy serialization (Rust-side only)
- CXX (dtolnay): zero-overhead Rust<->C++ bridge
- PyO3 + maturin: Python bindings with GIL release
- crossbeam: concurrent data structures
- bumpalo: arena allocator for compilation phase
- mimalloc: global allocator replacement

---

## 9. Key Architectural Decisions

### 9.1 C++/Rust Boundary

Rust does the "brain" (Plan Compiler): topology graph, algorithm selection, schedule optimization, buffer planning.
C++ does the "hands" (Custom Ops + Executor): ACL Runtime calls, kernel launches, HCCL P2P calls, stream management.

FFI boundary is narrow: PlanCache (Rust -> PyO3 -> Python -> C++ custom ops read parameters).

### 9.2 Standalone Executor C ABI (Fallback Path)

Only 3 functions: pgccl_compile, pgccl_execute, pgccl_instantiate.
Called infrequently (compile once, execute per-step but just passing serialized data).

### 9.3 torch Integration: Dual Mode

- Graph Mode (primary): pgccl graph optimization pass -> custom ops inside ACL Graph
- Compat Mode (fallback): ProcessGroupPGCCL implementing c10d::ProcessGroup

### 9.4 One-Sided Primitives (MSCCL++ lesson)

Put/Signal/Wait instead of Send/Recv. One-sided model unlocks all-pairs algorithms and reduces unnecessary synchronization. MSCCL++ showed 3.1x speedup for small-message AllReduce by switching from two-sided to one-sided.

### 9.5 Ascend Hardware Exploitation

MTE + AIV + Cube are physically separated engines (not shared like GPU SMs).
pgccl exploits this with pipelined inline processing:

```
MTE: load data to on-chip buffer (double-buffered)
AIV: quantize/dequantize in parallel with MTE
HCCS/SDMA: transfer compressed data in parallel with AIV
```

When pipeline is filled, transform latency is completely hidden.

---

## 10. High-Performance Rust Guide (for Plan Compiler)

### 10.1 Where Rust Performance Matters

NOT on the per-step hot path (that's C++/hardware).
Rust performance matters for three capability unlocks:
1. Deeper search -> better plans -> indirectly faster every step
2. Sub-microsecond instantiation -> per-request JIT plan (inference killer feature)
3. Sub-millisecond recompile -> online adaptive plans

### 10.2 Priority Actions

| Priority | Technique                        | Benefit              | Cost     |
|:---------|:---------------------------------|:---------------------|:---------|
| 1        | CSR graph format                 | Cache-friendly traversal | Low   |
| 2        | Vec::with_capacity + buffer reuse | Eliminate hot-path allocs | Low |
| 3        | lto="fat" + codegen-units=1 + mimalloc | 5-20% free perf | Zero code |
| 4        | bumpalo arena allocator          | Zero fragmentation   | Medium   |
| 5        | criterion benchmarks             | Guard <1ms red line  | Low      |
| 6        | PGO                              | 5-15% more           | CI change|

### 10.3 What NOT to Do
- No manual SIMD (graph algorithm bottleneck is memory access, not arithmetic)
- No premature unsafe (safe Rust is fast enough for <1000 node graphs)
- No parallel graph algorithms (data dependencies too heavy)
- No inline assembly

---

## 11. v0.1 Scope

### 11.1 Deliverables

```
pgccl v0.1 scope
+-- Plan Compiler [Rust]
|   +-- 8-card HCCS topology (hardcoded)
|   +-- AllReduce Ring algorithm (single algorithm)
|   +-- Buffer planning + schedule optimization
|   +-- Parameterized plan template (inference)
|
+-- Primary Path: torchair graph pass
|   +-- Recognize AllReduce pattern
|   +-- Replace with pgccl custom op
|   +-- ACL Graph compatibility verification
|
+-- pgccl Custom Ops [C++ AscendC]
|   +-- pgccl_pipelined_allreduce (2x AllReduce pipeline)
|   +-- pgccl_quantized_allreduce (INT8 compressed)
|   +-- Register as torch ops, graph-capturable
|
+-- Fallback: Standalone Executor
|   +-- KV cache pipeline transfer (layer-level pipeline)
|
+-- Benchmarks (3 groups)
|   +-- Training: 2x AllReduce 256MB -> bandwidth-oriented, vs HCCL
|   +-- Inference: 2x AllReduce 16KB -> latency-oriented, vs HCCL
|   +-- KV transfer: pipeline vs naive Send/Recv (32 layers, 64MB/layer)
|
+-- NOT in v0.1
    +-- MC2 fusion pass
    +-- Multi-machine RoCE
    +-- General topology discovery
    +-- Multiple algorithm selection
    +-- MoE-specific ops
```

### 11.2 Progressive Expansion

```
v0.1: Single machine 8 cards (HCCS only)
v0.2: Small multi-machine 2-8 nodes (HCCS + RoCE), AIV-Direct exploration
v0.3: Large scale 64+ machines (HCCS + RoCE + UB), schedule synthesis
```

### 11.3 Success Criteria

- v0.1 pipelined AllReduce: total time for 2x AllReduce < HCCL's 2x individual AllReduce
- v0.1 quantized AllReduce: busBW equivalent or better with 50% less data transfer
- v0.1 KV pipeline: first-token-to-decode latency < naive sequential transfer
- All custom ops captured by ACL Graph without breaking graph mode

---

## 12. NCCL/HCCL Weaknesses (Opportunities for pgccl)

| NCCL/HCCL Weakness              | pgccl Approach                              |
|:---------------------------------|:--------------------------------------------|
| Per-op algorithm selection       | AOT plan compilation, zero runtime decision  |
| SM resource contention (GPU)     | MTE/AIV/Cube physical separation (Ascend)    |
| Proxy thread bottleneck          | Direct device communication (future)         |
| No cross-op optimization         | Plan sees entire communication graph         |
| Fixed algorithm set              | Programmable algorithms (MSCCL-style)        |
| Opaque cost model                | Transparent, calibratable cost model         |
| Graph search scalability issues  | Iterative topology discovery or pre-computed DB |

---

## 13. ACL Runtime Constraints

| Constraint                | Value                           | pgccl Design Impact                |
|:--------------------------|:--------------------------------|:------------------------------------|
| Stream limit              | ~2048/device, 248 reserved      | Stream pool, coordinate with HCCL   |
| Huge page threshold       | >1MB                            | Pre-allocate buffers >= 1MB         |
| Host-device memory cost   | Up to 30% of total time         | Eliminate D2H in execution path     |
| ACL Graph stream per comm | 1+ per communication domain     | Budget stream usage in plan         |
| AllToAll D2H              | Breaks ACL Graph capture        | Use standalone executor for AllToAll|
| HCCL op expansion mode    | AIV (recommended) vs AICPU      | Prefer AIV mode for lower overhead  |
