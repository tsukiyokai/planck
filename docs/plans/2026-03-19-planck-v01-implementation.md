# Planck v0.1 Implementation Plan

> For agentic workers: REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: Implement Planck v0.1 -- 8-card HCCS Ring AllReduce plan compiler, Python bindings, C++ custom ops, and torchair integration with benchmarks against HCCL.

Architecture: Rust Plan Compiler (topo -> cost -> algo -> sched -> fusion -> serialize) produces an ExecutionPlan consumed by C++ custom ops or standalone executor. Python bindings via PyO3 expose PlanCache. torchair graph pass replaces AllReduce patterns with Planck custom ops inside ACL Graph. Two delivery channels: standalone executor (Channel A) and ACL Graph integration (Channel B).

Tech Stack: Rust stable, C++20, Python 3.8+, maturin, CMake + corrosion, PyO3, AscendC (CANN SDK)

Environment Split:
- Phase A (Chunks 1-5): Rust + Python, fully testable on macOS/Linux without Ascend hardware
- Phase B (Chunks 6-7): C++ + AscendC, requires Ascend NPU + CANN SDK

---

## File Structure

```
planck/
├── Cargo.toml                          # workspace root
├── pyproject.toml                      # maturin config
├── rust-toolchain.toml                 # pin stable channel
├── crates/
│   ├── planck-core/
│   │   ├── Cargo.toml                  # zero external deps for v0.1
│   │   ├── benches/
│   │   │   └── compile_bench.rs        # criterion benchmarks
│   │   └── src/
│   │       ├── lib.rs                  # re-exports
│   │       ├── topo.rs                 # topology (8-card HCCS)
│   │       ├── cost.rs                 # alpha-beta cost model
│   │       ├── algo.rs                 # Ring AllReduce decomposition
│   │       ├── sched.rs               # pipeline scheduler + buffer planner
│   │       ├── plan.rs                # IR types + compile() + fusion + serialization
│   │       └── template.rs            # parameterized plan templates
│   └── planck-python/
│       ├── Cargo.toml                  # depends on planck-core + pyo3
│       └── src/
│           └── lib.rs                  # PyO3 module
├── csrc/
│   ├── CMakeLists.txt
│   ├── include/planck/
│   │   ├── plan.h                      # C struct mirrors of Rust types
│   │   ├── transport.h                 # transport abstract interface
│   │   └── executor.h                  # standalone executor interface
│   ├── transport/
│   │   ├── hccs.cpp                    # HCCS transport (HCCL P2P)
│   │   └── mock.cpp                    # mock transport for testing
│   ├── ops/
│   │   ├── pipelined_allreduce.cpp     # 2x AllReduce pipeline
│   │   ├── quantized_allreduce.cpp     # INT8 compressed AllReduce
│   │   └── kv_pipeline_transfer.cpp    # KV cache layer pipeline
│   ├── kernels/
│   │   ├── quantize_per_group.cpp      # AscendC kernel
│   │   └── reduce_add.cpp             # AscendC kernel
│   ├── executor/
│   │   └── engine.cpp                  # standalone plan executor
│   └── torch_binding.cpp              # TORCH_LIBRARY registration
├── python/planck/
│   ├── __init__.py
│   ├── plan_cache.py                   # PlanCache Python wrapper
│   ├── graph_pass.py                   # torchair pattern replacement
│   └── ops.py                          # FakeTensor registration
├── tests/
│   ├── test_plan_compile.py            # Python compiler tests
│   ├── test_custom_ops.py              # needs Ascend hardware
│   ├── test_graph_capture.py           # needs torch_npu
│   └── bench_vs_hccl.py               # needs Ascend hardware
├── docs/plans/
│   ├── 2026-03-19-planck-design.md
│   └── 2026-03-19-planck-v01-implementation.md  # THIS FILE
├── CLAUDE.md
├── readme
└── tasks/
    ├── todo.md
    └── lessons.md
```

## Dependency Map

```
Chunk 1: Skeleton ─────────────────────────┐
                                           v
Chunk 2: IR + Topo + Cost ────────────────>│
                                           v
Chunk 3: Algo + Sched + Fusion ───────────>│
                                           v
Chunk 4: Compile + Serialize + Template ──>│
                                           v
Chunk 5: PyO3 Bindings ──────────────────>│
              |                            │
              v (requires Ascend)          │
Chunk 6: C++ Execution Layer <─────────────┘
              |
              v
Chunk 7: torchair + Benchmarks
```

---

## Chunk 1: Project Skeleton & Build System

### Task 1: Cargo Workspace

Files:
- Create: `Cargo.toml`
- Create: `rust-toolchain.toml`
- Create: `crates/planck-core/Cargo.toml`
- Create: `crates/planck-core/src/lib.rs`
- Create: `crates/planck-python/Cargo.toml`
- Create: `crates/planck-python/src/lib.rs`

- [ ] Step 1: Create workspace root Cargo.toml

```toml
# Cargo.toml
[workspace]
members = ["crates/planck-core", "crates/planck-python"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[profile.release]
lto = "fat"
codegen-units = 1
```

- [ ] Step 2: Create rust-toolchain.toml

```toml
# rust-toolchain.toml
[toolchain]
channel = "stable"
```

- [ ] Step 3: Create planck-core crate

```toml
# crates/planck-core/Cargo.toml
[package]
name = "planck-core"
version.workspace = true
edition.workspace = true

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "compile_bench"
harness = false
```

```rust
// crates/planck-core/src/lib.rs
pub mod plan;
pub mod topo;
pub mod cost;
pub mod algo;
pub mod sched;
pub mod template;
```

- [ ] Step 4: Create planck-python crate skeleton

```toml
# crates/planck-python/Cargo.toml
[package]
name = "planck-python"
version.workspace = true
edition.workspace = true

[lib]
name = "_planck"
crate-type = ["cdylib"]

[dependencies]
planck-core = { path = "../planck-core" }
pyo3 = { version = "0.22", features = ["extension-module"] }
```

```rust
// crates/planck-python/src/lib.rs
use pyo3::prelude::*;

#[pymodule]
fn _planck(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

- [ ] Step 5: Create pyproject.toml

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.5"]
build-backend = "maturin"

[project]
name = "planck"
requires-python = ">=3.8"
dynamic = ["version"]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "planck._planck"
manifest-path = "crates/planck-python/Cargo.toml"
python-source = "python"
```

- [ ] Step 6: Create empty module files for all Rust source files

```rust
// crates/planck-core/src/plan.rs
// crates/planck-core/src/topo.rs
// crates/planck-core/src/cost.rs
// crates/planck-core/src/algo.rs
// crates/planck-core/src/sched.rs
// crates/planck-core/src/template.rs
// (all initially empty)
```

- [ ] Step 7: Verify cargo build

Run: `cargo build`
Expected: successful compilation, zero warnings

- [ ] Step 8: Commit

```bash
git add Cargo.toml rust-toolchain.toml crates/ pyproject.toml
git commit -m "chore: Cargo workspace + maturin skeleton"
```

---

## Chunk 2: Plan IR Types & Topology (Rust)

### Task 2: Plan IR Types (plan.rs)

The core data types that define what a Plan IS. These are repr(C) for
direct serialization to C packed structs (no serde needed).

Files:
- Create: `crates/planck-core/src/plan.rs`

- [ ] Step 1: Write tests for IR types

```rust
// At bottom of plan.rs
#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn header_is_32_bytes() {
        assert_eq!(mem::size_of::<PlanHeader>(), 32);
    }

    #[test]
    fn buf_entry_is_12_bytes() {
        assert_eq!(mem::size_of::<BufEntry>(), 12);
    }

    #[test]
    fn op_entry_is_16_bytes() {
        assert_eq!(mem::size_of::<OpEntry>(), 16);
    }

    #[test]
    fn header_magic() {
        let h = PlanHeader::new(8, 0, 1, 0, 0);
        assert_eq!(h.magic, PLAN_MAGIC);
        assert_eq!(h.version, PLAN_VERSION);
    }

    #[test]
    fn opcode_values() {
        assert_eq!(Opcode::Put as u8, 0);
        assert_eq!(Opcode::WaitReducePut as u8, 7);
        assert_eq!(Opcode::Noop as u8, 8);
    }

    #[test]
    fn serialize_roundtrip() {
        let plan = ExecutionPlan {
            header: PlanHeader::new(8, 3, 2, 1, 4),
            buffers: vec![
                BufEntry { pool: BufPool::Input as u32, offset: 0, size: 1024 },
                BufEntry { pool: BufPool::Scratch as u32, offset: 0, size: 512 },
            ],
            ops: vec![
                OpEntry::new(Opcode::Put, 0, 0, 1, 1, ReduceOp::Sum, 0, 1),
                OpEntry::new(Opcode::Wait, 0, 0, 0, 7, ReduceOp::Sum, 1, 0),
            ],
        };
        let bytes = plan.serialize();
        let restored = ExecutionPlan::deserialize(&bytes).unwrap();
        assert_eq!(restored.header.num_ranks, 8);
        assert_eq!(restored.header.my_rank, 3);
        assert_eq!(restored.buffers.len(), 2);
        assert_eq!(restored.ops.len(), 2);
        assert_eq!(restored.ops[0].opcode, Opcode::Put as u8);
        assert_eq!(restored.ops[1].opcode, Opcode::Wait as u8);
    }
}
```

- [ ] Step 2: Run tests, verify they fail

Run: `cargo test -p planck-core -- plan`
Expected: compile error (types not defined yet)

- [ ] Step 3: Implement IR types

Key types to implement:

```rust
pub const PLAN_MAGIC: u32 = 0x4B4E_4C50;  // "PLNK" little-endian
pub const PLAN_VERSION: u16 = 1;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Put = 0,
    Signal,
    Wait,
    LocalCopy,
    LocalReduce,
    PutWithSignal,   // fused: Put + Signal
    WaitReduceCopy,  // fused: Wait + Reduce + Copy
    WaitReducePut,   // fused: Wait + Reduce + Put
    Noop,            // sync point
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum = 0, Max, Min }

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufPool { Scratch = 0, Input, Output }

// ---- Serialized (repr(C)) structures ----

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PlanHeader {
    pub magic: u32,          // PLAN_MAGIC
    pub version: u16,        // PLAN_VERSION
    pub num_ops: u16,
    pub num_buffers: u16,
    pub num_streams: u8,
    pub num_events: u8,
    pub num_ranks: u16,
    pub my_rank: u16,
    pub flags: u32,
    pub _reserved: [u8; 12],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BufEntry {
    pub pool: u32,     // BufPool as u32
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OpEntry {
    pub opcode: u8,
    pub stream_id: u8,
    pub reduce_op: u8,
    pub flags: u8,
    pub src_buf: u16,      // index into buffers[]
    pub dst_buf: u16,      // index into buffers[]
    pub dst_rank: u16,
    pub wait_event: u16,   // 0 = no wait
    pub signal_event: u16,
    pub _pad: u16,
}

pub struct ExecutionPlan {
    pub header: PlanHeader,
    pub buffers: Vec<BufEntry>,
    pub ops: Vec<OpEntry>,
}
```

Implement `PlanHeader::new()`, `OpEntry::new()`, `ExecutionPlan::serialize()`,
`ExecutionPlan::deserialize()`.

Serialization format: header bytes ++ buffer entries bytes ++ op entries bytes.
Use `unsafe { std::slice::from_raw_parts(...) }` for repr(C) -> byte slice conversion.
Deserialization: validate magic + version, reinterpret byte slices back to structs.

- [ ] Step 4: Run tests, verify they pass

Run: `cargo test -p planck-core -- plan`
Expected: all 6 tests pass

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/plan.rs
git commit -m "feat: Plan IR types with C-compatible serialization"
```

### Task 3: Topology (topo.rs)

8-card HCCS topology, hardcoded for v0.1. Each card connected to all
others via HCCS links with uniform bandwidth and latency.

Files:
- Create: `crates/planck-core/src/topo.rs`

- [ ] Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hccs_8card_basics() {
        let topo = Topology::hccs_8card();
        assert_eq!(topo.num_ranks, 8);
        // 8 cards, all-to-all = 8*7/2 = 28 undirected links = 56 directed
        assert_eq!(topo.links.len(), 56);
    }

    #[test]
    fn hccs_8card_ring_neighbors() {
        let topo = Topology::hccs_8card();
        // Ring: 0->1->2->...->7->0
        let ring = topo.ring_order();
        assert_eq!(ring.len(), 8);
        // Each rank's next/prev should be valid
        for i in 0..8 {
            let next = ring[(i + 1) % 8];
            assert!(topo.has_link(ring[i], next));
        }
    }

    #[test]
    fn link_properties() {
        let topo = Topology::hccs_8card();
        let link = topo.get_link(0, 1).unwrap();
        assert_eq!(link.transport, TransportType::Hccs);
        assert!(link.bandwidth_gbps > 0.0);
        assert!(link.latency_us > 0.0);
    }
}
```

- [ ] Step 2: Run tests, verify fail

Run: `cargo test -p planck-core -- topo`
Expected: compile error

- [ ] Step 3: Implement Topology

```rust
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType { Hccs = 0, Roce, Shm }

#[derive(Debug, Clone)]
pub struct Link {
    pub src: usize,
    pub dst: usize,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
    pub transport: TransportType,
}

#[derive(Debug, Clone)]
pub struct Topology {
    pub num_ranks: usize,
    pub links: Vec<Link>,
}
```

Key methods:
- `Topology::hccs_8card()`: create 56 directed links (all-to-all), HCCS bandwidth ~30 GB/s per link, latency ~1.5us
- `Topology::has_link(src, dst) -> bool`
- `Topology::get_link(src, dst) -> Option<&Link>`
- `Topology::ring_order() -> Vec<usize>`: returns [0, 1, 2, ..., 7] (simple sequential ring for v0.1)

Note on HCCS bandwidth: Atlas 800T A2 has 56 GB/s per HCCS port, 3-4 ports per die.
For v0.1, use 30 GB/s as a conservative per-link number. Mark as calibratable constant.

- [ ] Step 4: Run tests, verify pass

Run: `cargo test -p planck-core -- topo`
Expected: all 3 tests pass

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/topo.rs
git commit -m "feat: 8-card HCCS topology module"
```

### Task 4: Cost Model (cost.rs)

Hierarchical alpha-beta-gamma cost model. For v0.1, single-level (HCCS only).

Files:
- Create: `crates/planck-core/src/cost.rs`

- [ ] Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::Topology;

    #[test]
    fn cost_from_topology() {
        let topo = Topology::hccs_8card();
        let cost = CostModel::from_topology(&topo);
        assert!(cost.alpha_us > 0.0);   // startup latency
        assert!(cost.beta_us > 0.0);    // per-byte time
    }

    #[test]
    fn ring_allreduce_cost_scales_with_size() {
        let topo = Topology::hccs_8card();
        let cost = CostModel::from_topology(&topo);
        let t_small = cost.ring_allreduce(1024, 8);       // 1 KB
        let t_large = cost.ring_allreduce(256 << 20, 8);  // 256 MB
        assert!(t_large > t_small);
        // bandwidth-dominated: 256MB should be ~250000x slower than 1KB
        // but alpha adds a floor, so ratio < 250000
        assert!(t_large / t_small > 1000.0);
    }

    #[test]
    fn ring_cost_formula() {
        // Ring AllReduce: T = 2*(n-1)*alpha + 2*(n-1)/n * M * beta
        let cost = CostModel { alpha_us: 10.0, beta_us_per_byte: 0.001, gamma_us_per_byte: 0.0 };
        let t = cost.ring_allreduce(8000, 8); // 8000 bytes, 8 ranks
        let expected = 2.0 * 7.0 * 10.0 + 2.0 * 7.0 / 8.0 * 8000.0 * 0.001;
        assert!((t - expected).abs() < 1e-6);
    }
}
```

- [ ] Step 2: Run tests, verify fail

Run: `cargo test -p planck-core -- cost`

- [ ] Step 3: Implement CostModel

```rust
pub struct CostModel {
    pub alpha_us: f64,         // startup latency per step (us)
    pub beta_us_per_byte: f64, // per-byte transfer time (us/byte)
    pub gamma_us_per_byte: f64,// per-byte compute time (us/byte)
}
```

Key methods:
- `CostModel::from_topology(topo)`: extract alpha from link latency, beta from 1/bandwidth
- `CostModel::ring_allreduce(msg_size, num_ranks) -> f64`: standard Ring cost formula
  `T = 2*(n-1)*alpha + 2*(n-1)/n * msg_size * beta + (n-1)/n * msg_size * gamma`

- [ ] Step 4: Run tests, verify pass

Run: `cargo test -p planck-core -- cost`
Expected: all 3 tests pass

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/cost.rs
git commit -m "feat: alpha-beta-gamma cost model"
```

---

## Chunk 3: Ring Algorithm & Scheduler (Rust)

### Task 5: Ring AllReduce Algorithm (algo.rs)

Decomposes AllReduce into Ring ReduceScatter + AllGather steps.
Output: per-rank list of AlgoStep (abstract, before instruction generation).

Files:
- Create: `crates/planck-core/src/algo.rs`

- [ ] Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_allreduce_step_count() {
        let steps = ring_allreduce(8, 0);
        // RS: 7 steps + AG: 7 steps = 14
        assert_eq!(steps.len(), 14);
        assert_eq!(steps.iter().filter(|s| s.phase == Phase::ReduceScatter).count(), 7);
        assert_eq!(steps.iter().filter(|s| s.phase == Phase::AllGather).count(), 7);
    }

    #[test]
    fn ring_send_recv_ranks() {
        let steps = ring_allreduce(8, 3); // rank 3
        for step in &steps {
            // Ring: always send to (rank+1)%n, receive from (rank-1+n)%n
            assert_eq!(step.dst_rank, 4); // (3+1)%8
            assert_eq!(step.src_rank, 2); // (3-1+8)%8
        }
    }

    #[test]
    fn ring_rs_chunk_indices() {
        // Rank 0 in RS phase: sends chunk[0], chunk[7], chunk[6], ...
        // Receives chunk[7], chunk[6], chunk[5], ...
        let steps = ring_allreduce(8, 0);
        let rs_steps: Vec<_> = steps.iter()
            .filter(|s| s.phase == Phase::ReduceScatter)
            .collect();
        // Step 0: send chunk[(0-0)%8]=0, recv chunk[(0-0-1+8)%8]=7
        assert_eq!(rs_steps[0].send_chunk, 0);
        assert_eq!(rs_steps[0].recv_chunk, 7);
    }

    #[test]
    fn all_chunks_covered() {
        // After RS, each rank should have touched all chunks
        for rank in 0..8u16 {
            let steps = ring_allreduce(8, rank);
            let rs_recv: std::collections::HashSet<_> = steps.iter()
                .filter(|s| s.phase == Phase::ReduceScatter)
                .map(|s| s.recv_chunk)
                .collect();
            // Rank receives 7 distinct chunks (all except its own final chunk)
            assert_eq!(rs_recv.len(), 7);
        }
    }
}
```

- [ ] Step 2: Run tests, verify fail

Run: `cargo test -p planck-core -- algo`

- [ ] Step 3: Implement Ring AllReduce

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase { ReduceScatter, AllGather }

#[derive(Debug, Clone)]
pub struct AlgoStep {
    pub phase: Phase,
    pub step: u16,        // step index within phase (0..n-2)
    pub send_chunk: u16,  // data chunk index to send
    pub recv_chunk: u16,  // data chunk index to receive
    pub dst_rank: u16,    // send to
    pub src_rank: u16,    // receive from
    pub needs_reduce: bool,
}

/// Generate Ring AllReduce steps for given rank.
/// Ring order: 0 -> 1 -> 2 -> ... -> n-1 -> 0
pub fn ring_allreduce(num_ranks: u16, my_rank: u16) -> Vec<AlgoStep>
```

Ring AllReduce chunk index formulas:
- ReduceScatter step k: send chunk `(my_rank - k + n) % n`, recv chunk `(my_rank - k - 1 + n) % n`
- AllGather step k: send chunk `(my_rank - k + 1 + n) % n`, recv chunk `(my_rank - k + n) % n`
- RS: `needs_reduce = true`, AG: `needs_reduce = false`

- [ ] Step 4: Run tests, verify pass

Run: `cargo test -p planck-core -- algo`
Expected: all 4 tests pass

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/algo.rs
git commit -m "feat: Ring AllReduce algorithm decomposition"
```

### Task 6: Pipeline Scheduler (sched.rs)

Takes AlgoSteps and generates concrete OpEntry instructions with:
- Pipeline chunking (C chunks, each on its own stream)
- Buffer allocation (input/output/scratch for each pipeline chunk)
- Double-buffered receive for latency hiding

Files:
- Create: `crates/planck-core/src/sched.rs`

- [ ] Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo;

    #[test]
    fn schedule_produces_ops() {
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, 256 * 1024 * 1024, 4); // 256MB, 4 pipeline chunks
        assert!(!sched.ops.is_empty());
        assert!(!sched.buffers.is_empty());
    }

    #[test]
    fn schedule_uses_c_streams() {
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, 256 * 1024 * 1024, 4);
        let max_stream = sched.ops.iter().map(|op| op.stream_id).max().unwrap();
        assert_eq!(max_stream, 3); // 4 streams: 0,1,2,3
    }

    #[test]
    fn schedule_buffer_sizes() {
        let msg_size: usize = 256 << 20; // 256MB
        let chunks = 4;
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, msg_size, chunks);

        // Each pipeline chunk processes msg_size/chunks bytes
        // Data further divided into 8 ring chunks of msg_size/(chunks*8) each
        let chunk_size = msg_size / chunks;
        let ring_piece = chunk_size / 8;

        // Should have recv buffers (double-buffered) for each pipeline chunk
        let scratch_bufs: Vec<_> = sched.buffers.iter()
            .filter(|b| b.pool == crate::plan::BufPool::Scratch as u32)
            .collect();
        // At least C*2 scratch buffers (double buffer per pipeline chunk)
        assert!(scratch_bufs.len() >= chunks * 2);
        for b in &scratch_bufs {
            assert_eq!(b.size as usize, ring_piece);
        }
    }

    #[test]
    fn schedule_op_sequence_per_stream() {
        let steps = algo::ring_allreduce(8, 0);
        let sched = schedule(&steps, 8192, 1); // small, 1 chunk = no pipeline

        // For 1 pipeline chunk: 7 RS steps + 7 AG steps = 14 ring steps
        // Each RS step -> Put + Signal + Wait + LocalReduce = 4 ops (before fusion)
        // Each AG step -> Put + Signal + Wait + LocalCopy = 4 ops
        // Total: 14 * 4 = 56 ops
        let stream0_ops: Vec<_> = sched.ops.iter()
            .filter(|op| op.stream_id == 0)
            .collect();
        assert_eq!(stream0_ops.len(), 56);
    }
}
```

- [ ] Step 2: Run tests, verify fail

Run: `cargo test -p planck-core -- sched`

- [ ] Step 3: Implement scheduler

```rust
pub struct ScheduleResult {
    pub ops: Vec<OpEntry>,
    pub buffers: Vec<BufEntry>,
    pub num_streams: u8,
}

/// Pipeline-schedule AlgoSteps into concrete OpEntry instructions.
///
/// For each pipeline chunk c (stream c):
///   For each AlgoStep:
///     RS step -> Put(send_chunk) + Signal(dst) + Wait(src) + LocalReduce(recv -> accum)
///     AG step -> Put(send_chunk) + Signal(dst) + Wait(src) + LocalCopy(recv -> output)
pub fn schedule(steps: &[AlgoStep], msg_size: usize, pipeline_chunks: usize) -> ScheduleResult
```

Buffer layout:
- Input buffers: point into user tensor, offset by pipeline chunk
- Output buffers: same as input (in-place AllReduce) or separate
- Scratch buffers: 2 per pipeline chunk (double-buffered receive)
  - Size: `msg_size / (pipeline_chunks * num_ranks)` each
  - Alternate between buf_a and buf_b across ring steps

The scheduler assigns: `op.stream_id = pipeline_chunk_index`

- [ ] Step 4: Run tests, verify pass

Run: `cargo test -p planck-core -- sched`
Expected: all 4 tests pass

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/sched.rs
git commit -m "feat: pipeline scheduler with double-buffered receive"
```

### Task 7: Fusion Pass & Compilation Pipeline (plan.rs additions)

Add event assignment, instruction fusion, and the top-level `compile()` function
to plan.rs.

Files:
- Modify: `crates/planck-core/src/plan.rs`

- [ ] Step 1: Write tests

```rust
// Add to plan.rs tests module

#[test]
fn fusion_put_signal() {
    let ops = vec![
        OpEntry::new(Opcode::Put, 0, 0, 1, 1, ReduceOp::Sum, 0, 0),
        OpEntry::new(Opcode::Signal, 0, 0, 0, 1, ReduceOp::Sum, 0, 0),
    ];
    let fused = fuse(ops);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].opcode, Opcode::PutWithSignal as u8);
}

#[test]
fn fusion_wait_reduce_put() {
    let ops = vec![
        OpEntry::new(Opcode::Wait, 0, 0, 0, 7, ReduceOp::Sum, 0, 0),
        OpEntry::new(Opcode::LocalReduce, 0, 0, 1, 0, ReduceOp::Sum, 0, 0),
        OpEntry::new(Opcode::Put, 0, 1, 2, 1, ReduceOp::Sum, 0, 0),
    ];
    let fused = fuse(ops);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].opcode, Opcode::WaitReducePut as u8);
}

#[test]
fn fusion_preserves_unfusable() {
    let ops = vec![
        OpEntry::new(Opcode::Noop, 0, 0, 0, 0, ReduceOp::Sum, 0, 0),
        OpEntry::new(Opcode::Put, 0, 0, 1, 1, ReduceOp::Sum, 0, 0),
    ];
    let fused = fuse(ops);
    assert_eq!(fused.len(), 2); // no fusion possible
}

#[test]
fn compile_produces_valid_plan() {
    use crate::topo::Topology;

    let topo = Topology::hccs_8card();
    let req = CompileRequest {
        collective: Collective::AllReduce,
        msg_size: 256 << 20,
        reduce_op: ReduceOp::Sum,
        num_ranks: 8,
        my_rank: 0,
        pipeline_chunks: 4,
    };
    let plan = compile(&req, &topo);
    assert_eq!(plan.header.magic, PLAN_MAGIC);
    assert_eq!(plan.header.num_ranks, 8);
    assert_eq!(plan.header.my_rank, 0);
    assert!(plan.ops.len() > 0);
    assert!(plan.buffers.len() > 0);
    // After fusion, op count should be < 56 * 4 (4 chunks * 56 unfused ops)
    assert!(plan.ops.len() < 56 * 4);
}
```

- [ ] Step 2: Run tests, verify fail

Run: `cargo test -p planck-core -- plan`

- [ ] Step 3: Implement fusion and compile

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collective { AllReduce }

pub struct CompileRequest {
    pub collective: Collective,
    pub msg_size: usize,
    pub reduce_op: ReduceOp,
    pub num_ranks: usize,
    pub my_rank: usize,
    pub pipeline_chunks: usize,
}

/// Fuse adjacent instructions within each stream.
/// Patterns:
///   Put + Signal -> PutWithSignal
///   Wait + LocalReduce + Put -> WaitReducePut
///   Wait + LocalReduce + LocalCopy -> WaitReduceCopy
pub fn fuse(ops: Vec<OpEntry>) -> Vec<OpEntry>

/// Top-level compilation pipeline.
/// topo -> cost -> algo -> sched -> event_assign -> fuse -> ExecutionPlan
pub fn compile(req: &CompileRequest, topo: &Topology) -> ExecutionPlan
```

Fusion algorithm: sliding window over ops within same stream.
Match patterns greedily (longest match first: 3-op > 2-op).
When a pattern matches, replace with fused opcode, merge src/dst/rank fields.

Event assignment: for v0.1 with independent pipeline chunks (no cross-chunk
buffer reuse), no inter-stream events needed. Set `wait_event = 0`,
`signal_event = 0` for all ops. Events become relevant in v0.2 when
cross-chunk buffer reuse is added.

- [ ] Step 4: Run tests, verify pass

Run: `cargo test -p planck-core -- plan`
Expected: all tests pass (both new fusion tests and previous IR tests)

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/plan.rs
git commit -m "feat: instruction fusion pass + compile() pipeline"
```

---

## Chunk 4: Serialization, Templates & End-to-End Verification (Rust)

### Task 8: End-to-End Simulation Test

The ultimate correctness test: compile a plan, simulate its execution on
test data, verify the result matches a reference AllReduce.

This test generates plans for ALL 8 ranks and simulates the full ring
communication by executing instructions across ranks.

Files:
- Modify: `crates/planck-core/src/plan.rs` (add simulation test)

- [ ] Step 1: Write simulation test

```rust
#[test]
fn simulate_ring_allreduce() {
    let topo = Topology::hccs_8card();
    let n: usize = 8;
    let msg_size: usize = 8 * 8 * std::mem::size_of::<f32>(); // 8 chunks of 8 f32s = 256 bytes

    // Compile plans for all 8 ranks
    let plans: Vec<_> = (0..n).map(|rank| {
        compile(&CompileRequest {
            collective: Collective::AllReduce,
            msg_size,
            reduce_op: ReduceOp::Sum,
            num_ranks: n,
            my_rank: rank,
            pipeline_chunks: 1, // simple, no pipeline for this test
        }, &topo)
    }).collect();

    // Initialize per-rank data: rank r has [r+1, r+1, ...] (64 f32s)
    let num_f32s = msg_size / std::mem::size_of::<f32>();
    let mut data: Vec<Vec<f32>> = (0..n)
        .map(|r| vec![(r + 1) as f32; num_f32s])
        .collect();

    // Expected result: sum of all ranks = [1+2+...+8, ...] = [36.0, ...]
    let expected = vec![36.0f32; num_f32s];

    // Simulate execution (simplified: process ops in step order)
    // This requires a multi-rank simulator that handles Put/Signal/Wait
    // For v0.1, implement a minimal simulator in the test module
    let result = simulate_plans(&plans, &mut data, n);

    for rank in 0..n {
        assert_eq!(result[rank], expected, "rank {} mismatch", rank);
    }
}
```

The simulator needs to:
1. Process ops from all ranks in lockstep (by stream and step)
2. `Put`: copy data from src rank's buffer to dst rank's receive buffer
3. `Signal`/`Wait`: synchronization (no-op in simulation, ops are ordered)
4. `LocalReduce`: element-wise sum of recv buffer into accumulator
5. `LocalCopy`: copy recv buffer to output position
6. Fused ops: perform the combined operations

- [ ] Step 2: Implement minimal simulator (test-only code)

Create a `simulate_plans()` function inside `#[cfg(test)]`.
It interprets OpEntry instructions and moves f32 data between rank buffers.

This is the most complex test but validates the entire compiler pipeline end-to-end.

- [ ] Step 3: Run simulation, verify pass

Run: `cargo test -p planck-core -- simulate_ring_allreduce`
Expected: PASS (all ranks produce correct AllReduce result)

- [ ] Step 4: Commit

```bash
git add crates/planck-core/src/plan.rs
git commit -m "test: end-to-end Ring AllReduce simulation"
```

### Task 9: Parameterized Templates (template.rs)

Templates freeze the instruction graph but leave certain op fields as
parameterized slots (e.g., buffer offset depends on dynamic batch size).
Instantiation evaluates slot expressions to produce a concrete plan.

Files:
- Create: `crates/planck-core/src/template.rs`

- [ ] Step 1: Write tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::*;
    use crate::topo::Topology;

    #[test]
    fn template_creation() {
        let topo = Topology::hccs_8card();
        let req = CompileRequest {
            collective: Collective::AllReduce,
            msg_size: 1024, // placeholder
            reduce_op: ReduceOp::Sum,
            num_ranks: 8,
            my_rank: 0,
            pipeline_chunks: 1,
        };
        let plan = compile(&req, &topo);
        let template = PlanTemplate::from_plan(plan, vec![ParamSlot::MsgSize]);
        assert_eq!(template.param_slots.len(), 1);
        assert!(template.frozen_ops.len() > 0);
    }

    #[test]
    fn template_instantiation() {
        let topo = Topology::hccs_8card();
        let req = CompileRequest {
            collective: Collective::AllReduce,
            msg_size: 1024,
            reduce_op: ReduceOp::Sum,
            num_ranks: 8,
            my_rank: 0,
            pipeline_chunks: 1,
        };
        let plan = compile(&req, &topo);
        let template = PlanTemplate::from_plan(plan, vec![ParamSlot::MsgSize]);

        // Instantiate with 2048 bytes
        let plan_2k = template.instantiate(&[2048]);
        // Buffer sizes should scale with msg_size
        let total_buf: u32 = plan_2k.buffers.iter().map(|b| b.size).sum();
        assert!(total_buf > 0);

        // Instantiate with 4096 bytes
        let plan_4k = template.instantiate(&[4096]);
        let total_buf_4k: u32 = plan_4k.buffers.iter().map(|b| b.size).sum();
        // 4k plan should have roughly 2x the buffer of 2k plan
        assert!(total_buf_4k > total_buf);
    }

    #[test]
    fn instantiation_is_fast() {
        let topo = Topology::hccs_8card();
        let req = CompileRequest {
            collective: Collective::AllReduce,
            msg_size: 1024,
            reduce_op: ReduceOp::Sum,
            num_ranks: 8,
            my_rank: 0,
            pipeline_chunks: 4,
        };
        let plan = compile(&req, &topo);
        let template = PlanTemplate::from_plan(plan, vec![ParamSlot::MsgSize]);

        let start = std::time::Instant::now();
        for size in (1..1000).map(|i| i * 1024) {
            let _ = template.instantiate(&[size]);
        }
        let elapsed = start.elapsed();
        // 1000 instantiations should take < 1ms (sub-microsecond each)
        assert!(elapsed.as_millis() < 10, "too slow: {:?}", elapsed);
    }
}
```

- [ ] Step 2: Run tests, verify fail

Run: `cargo test -p planck-core -- template`

- [ ] Step 3: Implement PlanTemplate

```rust
#[derive(Debug, Clone, Copy)]
pub enum ParamSlot {
    MsgSize,      // buffer sizes scale with message size
    // v0.2: BatchSize, SeqLen, etc.
}

/// Frozen template: instruction graph structure is fixed,
/// buffer sizes/offsets are parameterized.
pub struct PlanTemplate {
    pub frozen_ops: Vec<OpEntry>,
    pub buffer_exprs: Vec<BufExpr>,  // how to compute each buffer's size/offset
    pub header: PlanHeader,
    pub param_slots: Vec<ParamSlot>,
}

/// Expression for computing buffer dimensions from parameters.
pub struct BufExpr {
    pub pool: u32,
    pub offset_scale: f64,  // offset = base_offset * (param / base_param)
    pub size_scale: f64,    // size = base_size * (param / base_param)
    pub base_offset: u32,
    pub base_size: u32,
    pub base_param: u64,    // the param value used during initial compilation
}

impl PlanTemplate {
    pub fn from_plan(plan: ExecutionPlan, slots: Vec<ParamSlot>) -> Self;
    pub fn instantiate(&self, params: &[u64]) -> ExecutionPlan;
}
```

Instantiation: iterate BufExpr table, evaluate each expression, fill into
new BufEntry. Copy frozen ops. Update header. O(num_buffers) work.

- [ ] Step 4: Run tests, verify pass

Run: `cargo test -p planck-core -- template`
Expected: all 3 tests pass

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/template.rs
git commit -m "feat: parameterized plan templates for inference JIT"
```

### Task 10: Criterion Benchmarks

Guard the <1ms compilation redline with criterion benchmarks.

Files:
- Create: `crates/planck-core/benches/compile_bench.rs`

- [ ] Step 1: Write benchmarks

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use planck_core::plan::*;
use planck_core::topo::Topology;

fn bench_compile(c: &mut Criterion) {
    let topo = Topology::hccs_8card();

    let mut group = c.benchmark_group("compile");
    for &(name, size, chunks) in &[
        ("256MB_4chunk", 256 << 20, 4),
        ("16KB_1chunk",  16 * 1024, 1),
        ("1MB_2chunk",   1 << 20,   2),
    ] {
        group.bench_with_input(BenchmarkId::new("ring_allreduce", name), &(), |b, _| {
            let req = CompileRequest {
                collective: Collective::AllReduce,
                msg_size: size,
                reduce_op: ReduceOp::Sum,
                num_ranks: 8,
                my_rank: 0,
                pipeline_chunks: chunks,
            };
            b.iter(|| compile(&req, &topo));
        });
    }
    group.finish();
}

fn bench_instantiate(c: &mut Criterion) {
    let topo = Topology::hccs_8card();
    let req = CompileRequest {
        collective: Collective::AllReduce,
        msg_size: 16 * 1024,
        reduce_op: ReduceOp::Sum,
        num_ranks: 8,
        my_rank: 0,
        pipeline_chunks: 1,
    };
    let plan = compile(&req, &topo);
    let template = planck_core::template::PlanTemplate::from_plan(
        plan, vec![planck_core::template::ParamSlot::MsgSize]);

    c.bench_function("instantiate_16KB", |b| {
        b.iter(|| template.instantiate(&[16 * 1024]));
    });
}

criterion_group!(benches, bench_compile, bench_instantiate);
criterion_main!(benches);
```

- [ ] Step 2: Run benchmarks

Run: `cargo bench -p planck-core`
Expected:
- compile: < 1ms for all sizes
- instantiate: < 1us

- [ ] Step 3: Commit

```bash
git add crates/planck-core/benches/
git commit -m "perf: criterion benchmarks for compile + instantiate"
```

---

## Chunk 5: PyO3 Bindings (Rust + Python)

### Task 11: PyO3 Module

Expose PlanCompiler and PlanCache to Python via PyO3.

Files:
- Modify: `crates/planck-python/src/lib.rs`

- [ ] Step 1: Write Python test first

```python
# tests/test_plan_compile.py
import pytest

def test_import():
    import planck
    assert hasattr(planck, '__version__')

def test_compile_allreduce():
    from planck import PlanCompiler
    compiler = PlanCompiler.hccs_8card()
    plan = compiler.compile_allreduce(
        msg_size=256 * 1024 * 1024,  # 256MB
        my_rank=0,
        reduce_op="sum",
        pipeline_chunks=4,
    )
    assert plan.num_ranks == 8
    assert plan.my_rank == 0
    assert plan.num_ops > 0
    assert plan.num_buffers > 0
    assert len(plan.to_bytes()) > 0

def test_plan_cache():
    from planck import PlanCache
    cache = PlanCache.hccs_8card()
    # First call compiles
    plan1 = cache.get_allreduce(msg_size=1024, my_rank=0)
    # Second call hits cache
    plan2 = cache.get_allreduce(msg_size=1024, my_rank=0)
    assert plan1.num_ops == plan2.num_ops

def test_template_instantiate():
    from planck import PlanCompiler
    compiler = PlanCompiler.hccs_8card()
    template = compiler.compile_template(
        my_rank=0,
        reduce_op="sum",
        pipeline_chunks=1,
    )
    plan_1k = template.instantiate(msg_size=1024)
    plan_2k = template.instantiate(msg_size=2048)
    assert plan_1k.num_ops == plan_2k.num_ops  # same structure
```

- [ ] Step 2: Implement PyO3 bindings

```rust
// crates/planck-python/src/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use planck_core::{plan, topo, template};

#[pyclass]
struct PlanCompiler {
    topo: topo::Topology,
}

#[pymethods]
impl PlanCompiler {
    #[staticmethod]
    fn hccs_8card() -> Self {
        Self { topo: topo::Topology::hccs_8card() }
    }

    fn compile_allreduce(&self, msg_size: usize, my_rank: usize,
                         reduce_op: &str, pipeline_chunks: usize) -> PyPlanView { ... }

    fn compile_template(&self, my_rank: usize, reduce_op: &str,
                        pipeline_chunks: usize) -> PyPlanTemplate { ... }
}

#[pyclass]
struct PlanCache {
    compiler: PlanCompiler,
    cache: std::collections::HashMap<(usize, usize), plan::ExecutionPlan>,
}

#[pyclass]
struct PyPlanView { /* fields mirroring plan header + serialized bytes */ }

#[pyclass]
struct PyPlanTemplate { inner: template::PlanTemplate }
```

Key: release GIL during compilation with `py.allow_threads(|| ...)`.

- [ ] Step 3: Create Python package structure

```python
# python/planck/__init__.py
from planck._planck import PlanCompiler, PlanCache, PyPlanView, PyPlanTemplate

__all__ = ["PlanCompiler", "PlanCache"]
```

- [ ] Step 4: Build and test

Run:
```bash
maturin develop
pytest tests/test_plan_compile.py -v
```
Expected: all 4 tests pass

- [ ] Step 5: Commit

```bash
git add crates/planck-python/ python/ tests/test_plan_compile.py
git commit -m "feat: PyO3 bindings for PlanCompiler and PlanCache"
```

---

## Chunk 6: C++ Execution Layer

> Requires Ascend NPU + CANN SDK for full compilation and testing.
> On macOS/non-Ascend: create headers, mock transport, and stub implementations.
> Mark hardware-dependent code with `#ifdef ASCEND_ENABLED`.

### Task 12: C Plan Struct Headers

C-compatible headers that mirror the Rust repr(C) types.
These allow C++ code to directly read serialized plans from Rust.

Files:
- Create: `csrc/include/planck/plan.h`

- [ ] Step 1: Write plan.h

```cpp
// csrc/include/planck/plan.h
#pragma once
#include <cstdint>

namespace planck {

constexpr uint32_t PLAN_MAGIC   = 0x4B4E4C50;  // "PLNK"
constexpr uint16_t PLAN_VERSION = 1;

enum class Opcode : uint8_t {
    Put = 0, Signal, Wait, LocalCopy, LocalReduce,
    PutWithSignal, WaitReduceCopy, WaitReducePut, Noop,
};

enum class ReduceOp : uint8_t { Sum = 0, Max, Min };
enum class BufPool  : uint32_t { Scratch = 0, Input, Output };

#pragma pack(push, 1)

struct PlanHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t num_ops;
    uint16_t num_buffers;
    uint8_t  num_streams;
    uint8_t  num_events;
    uint16_t num_ranks;
    uint16_t my_rank;
    uint32_t flags;
    uint8_t  _reserved[12];
};
static_assert(sizeof(PlanHeader) == 32);

struct BufEntry {
    uint32_t pool;
    uint32_t offset;
    uint32_t size;
};
static_assert(sizeof(BufEntry) == 12);

struct OpEntry {
    uint8_t  opcode;
    uint8_t  stream_id;
    uint8_t  reduce_op;
    uint8_t  flags;
    uint16_t src_buf;
    uint16_t dst_buf;
    uint16_t dst_rank;
    uint16_t wait_event;
    uint16_t signal_event;
    uint16_t _pad;
};
static_assert(sizeof(OpEntry) == 16);

#pragma pack(pop)

/// Zero-copy view into a serialized plan.
class PlanView {
public:
    explicit PlanView(const uint8_t* data, size_t len);
    const PlanHeader& header() const;
    const BufEntry*   buffers() const;
    const OpEntry*    ops() const;
    bool              valid() const;
private:
    const uint8_t* data_;
    size_t len_;
};

} // namespace planck
```

- [ ] Step 2: Write transport.h

```cpp
// csrc/include/planck/transport.h
#pragma once
#include <cstddef>
#include <cstdint>

namespace planck {

class Transport {
public:
    virtual ~Transport() = default;

    /// Async one-sided write to remote rank's buffer.
    virtual void put(uint16_t dst_rank,
                     const void* src, size_t src_size,
                     void* dst_remote, size_t dst_offset) = 0;

    /// Signal remote rank that data is ready.
    virtual void signal(uint16_t dst_rank) = 0;

    /// Wait for signal from remote rank.
    virtual void wait(uint16_t src_rank) = 0;

    /// Synchronize all pending operations on this transport.
    virtual void sync() = 0;
};

} // namespace planck
```

- [ ] Step 3: Write executor.h

```cpp
// csrc/include/planck/executor.h
#pragma once
#include "planck/plan.h"
#include "planck/transport.h"
#include <memory>

namespace planck {

/// Execute a pre-compiled plan. Zero runtime decisions.
class Executor {
public:
    struct Config {
        uint16_t my_rank;
        void*    input_buf;    // user's input tensor
        void*    output_buf;   // user's output tensor (can == input_buf)
        void*    scratch_buf;  // pre-allocated scratch memory
        size_t   scratch_size;
    };

    explicit Executor(std::shared_ptr<Transport> transport);

    /// Execute plan. Returns 0 on success.
    int execute(const PlanView& plan, const Config& config);

private:
    std::shared_ptr<Transport> transport_;
};

} // namespace planck
```

- [ ] Step 4: Commit

```bash
git add csrc/include/
git commit -m "feat: C++ plan/transport/executor headers"
```

### Task 13: Mock Transport & Plan Deserialization

Files:
- Create: `csrc/transport/mock.cpp`
- Create: `csrc/CMakeLists.txt`

- [ ] Step 1: Implement mock transport

```cpp
// csrc/transport/mock.cpp
#include "planck/transport.h"
#include <cstring>
#include <vector>
#include <unordered_map>

namespace planck {

/// In-process mock transport for testing.
/// All ranks share the same address space.
class MockTransport : public Transport {
public:
    explicit MockTransport(uint16_t num_ranks) : num_ranks_(num_ranks) {}

    void put(uint16_t dst_rank, const void* src, size_t size,
             void* dst_remote, size_t dst_offset) override {
        std::memcpy(static_cast<uint8_t*>(dst_remote) + dst_offset, src, size);
    }

    void signal(uint16_t) override {} // no-op in single-process mock
    void wait(uint16_t) override {}   // no-op in single-process mock
    void sync() override {}

private:
    uint16_t num_ranks_;
};

} // namespace planck
```

- [ ] Step 2: Implement PlanView

```cpp
// csrc/plan_view.cpp (or inline in plan.h)
namespace planck {

PlanView::PlanView(const uint8_t* data, size_t len)
    : data_(data), len_(len) {}

bool PlanView::valid() const {
    if (len_ < sizeof(PlanHeader)) return false;
    auto& h = header();
    if (h.magic != PLAN_MAGIC || h.version != PLAN_VERSION) return false;
    size_t expected = sizeof(PlanHeader)
                    + h.num_buffers * sizeof(BufEntry)
                    + h.num_ops * sizeof(OpEntry);
    return len_ >= expected;
}

const PlanHeader& PlanView::header() const {
    return *reinterpret_cast<const PlanHeader*>(data_);
}

const BufEntry* PlanView::buffers() const {
    return reinterpret_cast<const BufEntry*>(data_ + sizeof(PlanHeader));
}

const OpEntry* PlanView::ops() const {
    return reinterpret_cast<const OpEntry*>(
        data_ + sizeof(PlanHeader) + header().num_buffers * sizeof(BufEntry));
}

} // namespace planck
```

- [ ] Step 3: Write CMakeLists.txt (non-Ascend build)

```cmake
# csrc/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(planck-csrc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Headers
include_directories(include)

# Core library (non-Ascend: mock only)
add_library(planck_core STATIC
    transport/mock.cpp
)

# Plan deserialization test
add_executable(test_plan_view test_plan_view.cpp)
target_link_libraries(test_plan_view planck_core)
```

- [ ] Step 4: Write C++ plan deserialization test

```cpp
// csrc/test_plan_view.cpp
#include "planck/plan.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    using namespace planck;

    // Build a minimal plan in memory
    std::vector<uint8_t> buf(sizeof(PlanHeader) + sizeof(BufEntry) + sizeof(OpEntry));
    auto* h = reinterpret_cast<PlanHeader*>(buf.data());
    h->magic       = PLAN_MAGIC;
    h->version     = PLAN_VERSION;
    h->num_ops     = 1;
    h->num_buffers = 1;
    h->num_streams = 1;
    h->num_events  = 0;
    h->num_ranks   = 8;
    h->my_rank     = 0;
    h->flags       = 0;

    auto* b = reinterpret_cast<BufEntry*>(buf.data() + sizeof(PlanHeader));
    b->pool   = static_cast<uint32_t>(BufPool::Input);
    b->offset = 0;
    b->size   = 1024;

    auto* op = reinterpret_cast<OpEntry*>(buf.data() + sizeof(PlanHeader) + sizeof(BufEntry));
    op->opcode      = static_cast<uint8_t>(Opcode::Put);
    op->stream_id   = 0;
    op->src_buf     = 0;
    op->dst_buf     = 0;
    op->dst_rank    = 1;
    op->signal_event = 0;
    op->wait_event   = 0;

    PlanView view(buf.data(), buf.size());
    assert(view.valid());
    assert(view.header().num_ranks == 8);
    assert(view.buffers()[0].size == 1024);
    assert(view.ops()[0].opcode == static_cast<uint8_t>(Opcode::Put));

    std::printf("plan_view test: PASSED\n");
    return 0;
}
```

- [ ] Step 5: Build and test C++ (non-Ascend)

Run:
```bash
cd csrc && mkdir -p build && cd build
cmake .. && make
./test_plan_view
```
Expected: "plan_view test: PASSED"

- [ ] Step 6: Commit

```bash
git add csrc/
git commit -m "feat: C++ plan deserialization + mock transport"
```

### Task 14: Custom Op Stubs & torch_binding

Stub implementations for custom ops. These compile on any platform but
the actual communication calls are gated behind `#ifdef ASCEND_ENABLED`.

Files:
- Create: `csrc/ops/pipelined_allreduce.cpp`
- Create: `csrc/ops/quantized_allreduce.cpp`
- Create: `csrc/ops/kv_pipeline_transfer.cpp`
- Create: `csrc/torch_binding.cpp`

- [ ] Step 1: Write pipelined_allreduce stub

```cpp
// csrc/ops/pipelined_allreduce.cpp
#include "planck/plan.h"
#include "planck/transport.h"

namespace planck::ops {

/// Execute two AllReduce operations as a pipelined whole.
/// Plan compiler has pre-scheduled the interleaving of chunks.
///
/// Parameters from Plan Compiler (passed via custom op attributes):
///   plan_bytes: serialized ExecutionPlan
///   scratch_size: required scratch memory
void pipelined_allreduce(
    void* input_a, size_t size_a,   // first AllReduce tensor
    void* input_b, size_t size_b,   // second AllReduce tensor
    void* output_a, void* output_b,
    const uint8_t* plan_bytes, size_t plan_len,
    void* scratch, size_t scratch_size,
    Transport& transport)
{
    PlanView plan(plan_bytes, plan_len);
    if (!plan.valid()) return; // TODO: error handling

#ifdef ASCEND_ENABLED
    // Ascend implementation: execute plan on ACL streams
    // This will be filled in Phase B
#else
    // CPU reference implementation for testing
    // Simple memcpy + element-wise sum (no real communication)
#endif
}

} // namespace planck::ops
```

- [ ] Step 2: Write quantized_allreduce and kv_pipeline_transfer stubs (same pattern)

- [ ] Step 3: Write torch_binding.cpp

```cpp
// csrc/torch_binding.cpp
// Registers Planck custom ops with PyTorch's dispatcher.
// Requires libtorch; skip if not available.

#ifdef HAS_TORCH
#include <torch/library.h>

TORCH_LIBRARY(planck, m) {
    m.def("pipelined_allreduce(Tensor a, Tensor b, str plan_key) -> (Tensor, Tensor)");
    m.def("quantized_allreduce(Tensor input, str plan_key, int num_bits) -> Tensor");
    m.def("kv_pipeline_transfer(Tensor[] kv_layers, str plan_key) -> Tensor[]");
}

// NPU implementation registered via TORCH_LIBRARY_IMPL in Ascend-specific code
#endif
```

- [ ] Step 4: Commit

```bash
git add csrc/ops/ csrc/torch_binding.cpp
git commit -m "feat: custom op stubs + TORCH_LIBRARY registration"
```

### Task 15: Standalone Executor Engine

The standalone executor reads a serialized plan and executes it
instruction by instruction on a Transport.

Files:
- Create: `csrc/executor/engine.cpp`

- [ ] Step 1: Write executor

```cpp
// csrc/executor/engine.cpp
#include "planck/executor.h"
#include <cstring>

namespace planck {

Executor::Executor(std::shared_ptr<Transport> transport)
    : transport_(std::move(transport)) {}

int Executor::execute(const PlanView& plan, const Config& cfg) {
    if (!plan.valid()) return -1;

    const auto& h = plan.header();
    const auto* bufs = plan.buffers();
    const auto* ops  = plan.ops();

    // Resolve buffer pointers
    auto resolve_buf = [&](uint16_t idx) -> void* {
        const auto& b = bufs[idx];
        void* base = nullptr;
        switch (static_cast<BufPool>(b.pool)) {
            case BufPool::Input:   base = cfg.input_buf;   break;
            case BufPool::Output:  base = cfg.output_buf;  break;
            case BufPool::Scratch: base = cfg.scratch_buf;  break;
        }
        return static_cast<uint8_t*>(base) + b.offset;
    };

    // Fast path: sequential op execution (zero decisions)
    for (uint16_t i = 0; i < h.num_ops; ++i) {
        const auto& op = ops[i];
        switch (static_cast<Opcode>(op.opcode)) {
            case Opcode::Put:
                transport_->put(op.dst_rank,
                    resolve_buf(op.src_buf), bufs[op.src_buf].size,
                    resolve_buf(op.dst_buf), 0);
                break;
            case Opcode::Signal:
                transport_->signal(op.dst_rank);
                break;
            case Opcode::Wait:
                transport_->wait(op.dst_rank);
                break;
            case Opcode::LocalCopy:
                std::memcpy(resolve_buf(op.dst_buf),
                            resolve_buf(op.src_buf),
                            bufs[op.src_buf].size);
                break;
            case Opcode::LocalReduce: {
                // Element-wise sum (float32)
                auto* dst = static_cast<float*>(resolve_buf(op.dst_buf));
                auto* src = static_cast<const float*>(resolve_buf(op.src_buf));
                size_t n = bufs[op.src_buf].size / sizeof(float);
                for (size_t j = 0; j < n; ++j) dst[j] += src[j];
                break;
            }
            case Opcode::PutWithSignal:
                transport_->put(op.dst_rank,
                    resolve_buf(op.src_buf), bufs[op.src_buf].size,
                    resolve_buf(op.dst_buf), 0);
                transport_->signal(op.dst_rank);
                break;
            case Opcode::WaitReduceCopy: {
                transport_->wait(op.dst_rank);
                auto* dst = static_cast<float*>(resolve_buf(op.dst_buf));
                auto* src = static_cast<const float*>(resolve_buf(op.src_buf));
                size_t n = bufs[op.src_buf].size / sizeof(float);
                for (size_t j = 0; j < n; ++j) dst[j] += src[j];
                break;
            }
            case Opcode::WaitReducePut: {
                transport_->wait(op.dst_rank);
                auto* dst = static_cast<float*>(resolve_buf(op.dst_buf));
                auto* src = static_cast<const float*>(resolve_buf(op.src_buf));
                size_t n = bufs[op.src_buf].size / sizeof(float);
                for (size_t j = 0; j < n; ++j) dst[j] += src[j];
                transport_->put(op.dst_rank, dst, bufs[op.dst_buf].size,
                    resolve_buf(op.dst_buf), 0);
                transport_->signal(op.dst_rank);
                break;
            }
            case Opcode::Noop:
                break;
        }
    }
    transport_->sync();
    return 0;
}

} // namespace planck
```

- [ ] Step 2: Add executor to CMakeLists.txt and test

Add `executor/engine.cpp` to `planck_core` library.
Write a C++ test that: compile plan in Rust (via saved bytes) -> load in C++ -> execute on mock transport -> verify result.

For cross-language testing, save plan bytes to a file from Python/Rust test, load in C++ test.

- [ ] Step 3: Commit

```bash
git add csrc/executor/
git commit -m "feat: standalone plan executor engine"
```

---

## Chunk 7: torchair Integration & Benchmarks

> Requires Ascend NPU + CANN SDK + torch_npu + torchair.

### Task 16: torchair Graph Optimization Pass

Pattern-matches AllReduce operations in the graph and replaces them
with Planck custom ops that use pre-compiled plans.

Files:
- Create: `python/planck/graph_pass.py`
- Create: `python/planck/ops.py`

- [ ] Step 1: Write graph_pass.py

```python
# python/planck/graph_pass.py
"""
torchair graph optimization pass: replace AllReduce patterns
with Planck pipelined/quantized custom ops.

Usage:
    import planck
    planck.register_graph_pass()
    # Then compile model with torchair as usual
"""

def _pattern_two_allreduce(a, b):
    """Match two adjacent AllReduce ops on the same group."""
    import torch.distributed as dist
    ra = dist.all_reduce(a)
    rb = dist.all_reduce(b)
    return ra, rb

def _replacement_pipelined(a, b):
    """Replace with Planck pipelined AllReduce."""
    import torch
    return torch.ops.planck.pipelined_allreduce(a, b, "allreduce_pipeline")

def _pattern_large_allreduce(x):
    """Match a single large AllReduce (>1MB)."""
    import torch.distributed as dist
    return dist.all_reduce(x)

def _replacement_quantized(x):
    """Replace with Planck quantized AllReduce (INT8)."""
    import torch
    return torch.ops.planck.quantized_allreduce(x, "allreduce_quantized", 8)

def register_graph_pass():
    """Register Planck patterns with torchair."""
    try:
        from torchair import register_replacement
        register_replacement(_pattern_two_allreduce, _replacement_pipelined)
        register_replacement(_pattern_large_allreduce, _replacement_quantized)
    except ImportError:
        pass  # torchair not available, skip graph pass registration
```

- [ ] Step 2: Write ops.py (FakeTensor registration)

```python
# python/planck/ops.py
"""
FakeTensor implementations for Planck custom ops.
Required for torch.compile / ACL Graph capture to trace through custom ops.
"""
import torch
from torch.library import Library, impl

# Already defined schema in C++ TORCH_LIBRARY; register fake implementations here
try:
    @torch.library.register_fake("planck::pipelined_allreduce")
    def _(a, b, plan_key):
        return a.clone(), b.clone()

    @torch.library.register_fake("planck::quantized_allreduce")
    def _(input, plan_key, num_bits):
        return input.clone()

    @torch.library.register_fake("planck::kv_pipeline_transfer")
    def _(kv_layers, plan_key):
        return [t.clone() for t in kv_layers]
except Exception:
    pass  # torch not available or ops not registered
```

- [ ] Step 3: Update __init__.py

```python
# python/planck/__init__.py
from planck._planck import PlanCompiler, PlanCache

def register_graph_pass():
    from planck.graph_pass import register_graph_pass as _register
    _register()

__all__ = ["PlanCompiler", "PlanCache", "register_graph_pass"]
```

- [ ] Step 4: Write graph capture test (requires torch_npu)

```python
# tests/test_graph_capture.py
"""Test that Planck custom ops are capturable by ACL Graph."""
import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

def test_pipelined_allreduce_graph_capture():
    """Custom op should not break graph capture."""
    import planck
    planck.register_graph_pass()

    a = torch.randn(1024, device="npu")
    b = torch.randn(1024, device="npu")

    # Verify graph capture completes without error
    # (correctness tested separately with real communication)
    compiled = torch.compile(
        lambda x, y: torch.ops.planck.pipelined_allreduce(x, y, "test"),
        backend="torchair"
    )
    out_a, out_b = compiled(a, b)
    assert out_a.shape == a.shape
    assert out_b.shape == b.shape
```

- [ ] Step 5: Commit

```bash
git add python/planck/ tests/test_graph_capture.py
git commit -m "feat: torchair graph pass + FakeTensor registration"
```

### Task 17: Benchmarks

Three benchmark groups as specified in v0.1 scope.

Files:
- Create: `tests/bench_vs_hccl.py`

- [ ] Step 1: Write benchmark script

```python
# tests/bench_vs_hccl.py
"""
Planck v0.1 Benchmarks vs HCCL.
Requires 8x Ascend NPU.
Run: torchrun --nproc_per_node=8 tests/bench_vs_hccl.py
"""
import time
import argparse
import torch
import torch.distributed as dist

def bench_hccl_allreduce(size_bytes, warmup=10, iters=100):
    """Baseline: two sequential HCCL AllReduce calls."""
    t = torch.randn(size_bytes // 4, device="npu", dtype=torch.float32)
    for _ in range(warmup):
        dist.all_reduce(t)
        dist.all_reduce(t)
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
        dist.all_reduce(t)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms

def bench_planck_pipelined(size_bytes, warmup=10, iters=100):
    """Planck: pipelined AllReduce (two ops as one)."""
    import planck
    a = torch.randn(size_bytes // 4, device="npu", dtype=torch.float32)
    b = torch.randn(size_bytes // 4, device="npu", dtype=torch.float32)
    for _ in range(warmup):
        torch.ops.planck.pipelined_allreduce(a, b, "bench")
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.ops.planck.pipelined_allreduce(a, b, "bench")
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms

def main():
    dist.init_process_group("hccl")
    rank = dist.get_rank()

    results = []

    # Group 1: Training (large message, bandwidth-oriented)
    for label, size in [("256MB", 256 << 20), ("64MB", 64 << 20)]:
        t_hccl   = bench_hccl_allreduce(size)
        t_planck = bench_planck_pipelined(size)
        bw_hccl   = 2 * size / t_hccl * 1000 / 1e9   # GB/s (2x for 2 ops)
        bw_planck = 2 * size / t_planck * 1000 / 1e9
        results.append((f"Train {label}", t_hccl, t_planck, bw_planck / bw_hccl))

    # Group 2: Inference (small message, latency-oriented)
    for label, size in [("16KB", 16 * 1024), ("64KB", 64 * 1024)]:
        t_hccl   = bench_hccl_allreduce(size)
        t_planck = bench_planck_pipelined(size)
        results.append((f"Infer {label}", t_hccl, t_planck, t_hccl / t_planck))

    if rank == 0:
        print(f"{'Scenario':<16} {'HCCL(ms)':>10} {'Planck(ms)':>12} {'Speedup':>10}")
        print("=" * 52)
        for name, t_h, t_p, ratio in results:
            print(f"{name:<16} {t_h:>10.3f} {t_p:>12.3f} {ratio:>9.2f}x")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

- [ ] Step 2: Write KV transfer benchmark (separate file or section)

```python
# Group 3: KV cache transfer benchmark
# Compare: sequential HcclSend/Recv per layer vs Planck pipeline
def bench_kv_sequential(num_layers=32, layer_size=64*1024*1024):
    """Baseline: sequential Send/Recv for each KV layer."""
    layers = [torch.randn(layer_size // 4, device="npu") for _ in range(num_layers)]
    # ... sequential transfer timing ...

def bench_kv_planck_pipeline(num_layers=32, layer_size=64*1024*1024):
    """Planck: pipelined KV transfer across layers."""
    layers = [torch.randn(layer_size // 4, device="npu") for _ in range(num_layers)]
    # ... planck.kv_pipeline_transfer timing ...
```

- [ ] Step 3: Commit

```bash
git add tests/bench_vs_hccl.py
git commit -m "perf: benchmark suite vs HCCL (train/infer/kv-transfer)"
```

---

## Success Criteria Checklist

After all chunks are complete, verify:

- [ ] `cargo test -p planck-core` -- all Rust tests pass (including simulation)
- [ ] `cargo bench -p planck-core` -- compile <1ms, instantiate <1us
- [ ] `maturin develop && pytest tests/test_plan_compile.py` -- Python bindings work
- [ ] `cd csrc/build && ./test_plan_view` -- C++ plan deserialization works
- [ ] (Ascend) `pytest tests/test_graph_capture.py` -- ACL Graph capture works
- [ ] (Ascend) `torchrun --nproc_per_node=8 tests/bench_vs_hccl.py` -- performance targets met:
  - Pipelined AllReduce 256MB: total time < 2x individual HCCL AllReduce
  - Quantized AllReduce: equivalent busBW with 50% less data transfer
  - KV pipeline: first-token latency < sequential transfer

## Execution Order Summary

```
Phase A (macOS, no hardware dependency):
  Chunk 1 -> Chunk 2 -> Chunk 3 -> Chunk 4 -> Chunk 5
  ~17 tasks, ~85 steps

Phase B (Ascend NPU required):
  Chunk 6 -> Chunk 7
  ~6 tasks, ~20 steps
```

Estimated Rust LOC: ~1500 (planck-core) + ~200 (planck-python)
Estimated C++ LOC: ~500 (headers + executor + stubs)
Estimated Python LOC: ~300 (bindings + graph pass + benchmarks + tests)
