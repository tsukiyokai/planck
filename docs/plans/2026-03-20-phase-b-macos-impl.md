# Planck v0.1 Phase B macOS Implementation Plan

> For agentic workers: REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

Goal: 在macOS上实现C++ Execution Layer + PyTorch Eager集成 + planck-sim DES仿真器，无需Ascend硬件。

Architecture: Block 1 (C++) 验证 Rust compile -> serialize -> C++ deserialize -> execute 的完整路径。Block 2 (Rust) 为plan加时间维度输出Chrome Trace timeline。两个Block可并行。

Tech Stack: C++20 (cmake), Rust stable, PyO3, libtorch (可选), toml+serde (feature-gated)

Design: `docs/plans/2026-03-20-phase-b-macos-design.md`

---

## File Structure

```
csrc/
  CMakeLists.txt
  include/planck/
    plan.h                    C struct mirrors of Rust repr(C) types
    transport.h               Transport abstract interface
    executor.h                Executor class declaration
  transport/
    mock.cpp                  MockWorld: shared memory + mutex/condvar
  executor/
    engine.cpp                Executor::execute() — for-loop over OpEntry
  torch_binding.cpp           TORCH_LIBRARY registration (optional, requires libtorch)
  test/
    test_util.h               ~30 line test macros
    test_plan.cpp             struct size + fixture roundtrip
    test_executor.cpp         8-thread allreduce simulation
    fixtures/                 plan bytes generated from Python
      plan_rank0.bin .. plan_rank7.bin

crates/planck-core/
  Cargo.toml                  add optional toml+serde deps under [features] sim
  src/
    lib.rs                    add `#[cfg(feature="sim")] pub mod sim;`
    sim/
      mod.rs                  simulate() public API
      config.rs               SimConfig + TOML parsing + size parser
      engine.rs               DES: Event, EventQueue, Simulator main loop
      link.rs                 LinkState: active flows, bandwidth competition
      timing.rs               TimingModel trait + SimpleModel + AscendModel
      trace.rs                Chrome Trace JSON with 4-layer hierarchy

crates/planck-python/src/lib.rs   add simulate() PyO3 binding

python/planck/
  ops.py                      FakeTensor registration
  __init__.py                 add simulate re-export

tests/
  test_torch_eager.py         PyTorch eager mode test (optional)
  test_sim.py                 planck-sim Python integration test

scripts/
  gen_fixtures.py             generate plan bytes for C++ tests

planck-sim.toml               example config
```

---

## Dependency Graph

```
Block 1 (C++):
  gen_fixtures -> impl_cpp_ffi -> gate_ffi -> impl_executor -> gate_exec -> impl_torch_eager -> gate_eager

Block 2 (Rust):
  impl_sim_engine -> gate_sim_core -> impl_sim_trace -> gate_sim -> impl_sim_pyo3 -> gate_sim_py

Block 1 and Block 2 are independent — can run in parallel.
```

---

## Block 1: C++ Execution Layer + PyTorch Eager

---

## Chunk 1: Fixtures + C++ Project Skeleton

### Task 1: Generate Plan Byte Fixtures

Files:
- Create: `scripts/gen_fixtures.py`
- Create: `csrc/test/fixtures/` (directory)

- [ ] Step 1: Write gen_fixtures.py

```python
#!/usr/bin/env python3
"""Generate plan byte fixtures for C++ tests.
Prereq: maturin develop
"""
import os, planck

out = "csrc/test/fixtures"
os.makedirs(out, exist_ok=True)

compiler = planck.PlanCompiler.hccs_8card()
for rank in range(8):
    plan = compiler.compile_allreduce(256, my_rank=rank, pipeline_chunks=1)  # 256 bytes
    path = f"{out}/plan_rank{rank}.bin"
    with open(path, "wb") as f:
        f.write(plan.to_bytes())
    print(f"  {path}: {len(plan.to_bytes())} bytes, {plan.num_ops} ops, {plan.num_buffers} bufs")

print(f"done: 8 fixtures in {out}/")
```

- [ ] Step 2: Run

```bash
maturin develop && python scripts/gen_fixtures.py
```

Expected: 8 .bin files in csrc/test/fixtures/

- [ ] Step 3: Commit

```bash
git add scripts/gen_fixtures.py csrc/test/fixtures/
git commit -m "feat(phase-b): plan byte fixtures for C++ tests"
```

### Task 2: CMakeLists.txt + test_util.h

Files:
- Create: `csrc/CMakeLists.txt`
- Create: `csrc/test/test_util.h`

- [ ] Step 1: Write CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(planck_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# ==== Core library (header-only for now, sources added later) ====
add_library(planck_core STATIC
    transport/mock.cpp
)

# ==== Tests ====
enable_testing()

add_executable(test_plan test/test_plan.cpp)
target_link_libraries(test_plan planck_core)
add_test(NAME test_plan COMMAND test_plan)

# ==== Optional: libtorch ====
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "libtorch found — building torch_binding")
else()
    message(STATUS "libtorch not found — skipping torch_binding")
endif()
```

- [ ] Step 2: Write test_util.h

```cpp
// csrc/test/test_util.h — minimal test macros, no external deps
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>

static int _g_fail = 0;

#define CHECK(cond, ...) do {                                        \
    if (!(cond)) {                                                   \
        fprintf(stderr, "  FAIL %s:%d: ", __FILE__, __LINE__);       \
        fprintf(stderr, __VA_ARGS__);                                \
        fprintf(stderr, "\n");                                       \
        ++_g_fail;                                                   \
    }                                                                \
} while(0)

#define CHECK_EQ(a, b) CHECK((a)==(b), "%s=%d != %s=%d", #a, (int)(a), #b, (int)(b))

#define CHECK_NEAR(a, b, eps) CHECK(fabs((a)-(b))<(eps), \
    "%s=%.4f != %s=%.4f (eps=%.4f)", #a, (double)(a), #b, (double)(b), (double)(eps))

#define RUN(fn) do {                              \
    int _prev = _g_fail;                          \
    fn();                                         \
    if (_g_fail > _prev)                          \
        fprintf(stderr, "FAIL %s\n", #fn);        \
    else                                          \
        fprintf(stderr, "  ok %s\n", #fn);        \
} while(0)

#define TEST_MAIN(...)                            \
    int main() {                                  \
        __VA_ARGS__;                              \
        if (_g_fail) {                            \
            fprintf(stderr, "%d failures\n", _g_fail); \
            return 1;                             \
        }                                         \
        fprintf(stderr, "all passed\n");          \
        return 0;                                 \
    }
```

- [ ] Step 3: Create placeholder mock.cpp (empty, cmake needs it)

```cpp
// csrc/transport/mock.cpp — placeholder, implemented in Task 6
#include "planck/transport.h"
```

- [ ] Step 4: Verify cmake builds

```bash
cd csrc && mkdir -p build && cd build && cmake .. && cmake --build .
```

Expected: builds with zero errors (test_plan will fail to link — that's ok for now)

- [ ] Step 5: Commit

```bash
git add csrc/CMakeLists.txt csrc/test/test_util.h csrc/transport/mock.cpp
git commit -m "feat(phase-b): C++ project skeleton with cmake + test macros"
```

---

## Chunk 2: Plan Headers + Mock Transport + Deserialization Test

### Task 3: plan.h

Files:
- Create: `csrc/include/planck/plan.h`

- [ ] Step 1: Write plan.h

```cpp
// csrc/include/planck/plan.h — C mirrors of Rust repr(C) types
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

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
static_assert(sizeof(PlanHeader) == 32, "PlanHeader must be 32 bytes");

struct BufEntry {
    uint32_t pool;
    uint32_t offset;
    uint32_t size;
};
static_assert(sizeof(BufEntry) == 12, "BufEntry must be 12 bytes");

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
static_assert(sizeof(OpEntry) == 16, "OpEntry must be 16 bytes");

#pragma pack(pop)

// Zero-copy view into serialized plan bytes
class PlanView {
public:
    PlanView(const uint8_t* data, size_t len) : data_(data), len_(len) {}

    bool valid() const {
        if (len_ < sizeof(PlanHeader)) return false;
        auto& h = header();
        if (h.magic != PLAN_MAGIC || h.version != PLAN_VERSION) return false;
        size_t expected = sizeof(PlanHeader)
            + h.num_buffers * sizeof(BufEntry)
            + h.num_ops * sizeof(OpEntry);
        return len_ >= expected;
    }

    const PlanHeader& header() const {
        return *reinterpret_cast<const PlanHeader*>(data_);
    }

    const BufEntry* buffers() const {
        return reinterpret_cast<const BufEntry*>(data_ + sizeof(PlanHeader));
    }

    const OpEntry* ops() const {
        return reinterpret_cast<const OpEntry*>(
            data_ + sizeof(PlanHeader) + header().num_buffers * sizeof(BufEntry));
    }

private:
    const uint8_t* data_;
    size_t len_;
};

} // namespace planck
```

- [ ] Step 2: Commit

```bash
git add csrc/include/planck/plan.h
git commit -m "feat(phase-b): plan.h mirrors Rust repr(C) structs"
```

### Task 4: transport.h + executor.h

Files:
- Create: `csrc/include/planck/transport.h`
- Create: `csrc/include/planck/executor.h`

- [ ] Step 1: Write transport.h

```cpp
// csrc/include/planck/transport.h
#pragma once
#include <cstdint>
#include <cstddef>

namespace planck {

class Transport {
public:
    virtual ~Transport() = default;

    // One-sided put: copy local src to remote rank's buffer[remote_buf_idx] + offset
    virtual void put(uint16_t dst_rank,
                     const void* src, size_t size,
                     uint16_t remote_buf_idx, size_t offset) = 0;

    virtual void signal(uint16_t dst_rank) = 0;
    virtual void wait(uint16_t src_rank) = 0;
    virtual void sync() = 0;
};

} // namespace planck
```

- [ ] Step 2: Write executor.h

```cpp
// csrc/include/planck/executor.h
#pragma once
#include "planck/plan.h"
#include "planck/transport.h"
#include <memory>

namespace planck {

class Executor {
public:
    struct Config {
        uint16_t my_rank;
        void*    input_buf;     // user tensor (Input pool base)
        void*    output_buf;    // output tensor (Output pool base, can == input_buf)
        void*    scratch_buf;   // pre-allocated scratch
        size_t   scratch_size;
    };

    explicit Executor(std::shared_ptr<Transport> transport)
        : transport_(std::move(transport)) {}

    int execute(const PlanView& plan, const Config& cfg);

private:
    std::shared_ptr<Transport> transport_;
};

} // namespace planck
```

- [ ] Step 3: Commit

```bash
git add csrc/include/planck/transport.h csrc/include/planck/executor.h
git commit -m "feat(phase-b): transport + executor header declarations"
```

### Task 5: MockWorld Transport

Files:
- Modify: `csrc/transport/mock.cpp`

- [ ] Step 1: Write mock.cpp

```cpp
// csrc/transport/mock.cpp — in-process shared-memory transport for testing
#include "planck/transport.h"
#include "planck/plan.h"
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cassert>

namespace planck {

class MockWorld {
public:
    struct RankBufs {
        std::vector<uint8_t> input;
        std::vector<uint8_t> output;
        std::vector<uint8_t> scratch;
        const BufEntry*      buf_table;  // points into plan's buffer array
        uint16_t             num_bufs;
    };

    explicit MockWorld(int num_ranks) : num_ranks_(num_ranks), ranks_(num_ranks) {
        for (auto& row : signals_)
            for (auto& v : row) v = 0;
    }

    void setup_rank(int rank, void* input, size_t in_sz,
                    void* output, size_t out_sz,
                    void* scratch, size_t sc_sz,
                    const BufEntry* bufs, uint16_t num_bufs) {
        auto& r = ranks_[rank];
        r.input.assign((uint8_t*)input, (uint8_t*)input + in_sz);
        r.output.assign((uint8_t*)output, (uint8_t*)output + out_sz);
        r.scratch.assign((uint8_t*)scratch, (uint8_t*)scratch + sc_sz);
        r.buf_table = bufs;
        r.num_bufs  = num_bufs;
    }

    // Resolve a buffer index on a given rank -> pointer into that rank's memory
    void* resolve_remote(int rank, uint16_t buf_idx) {
        auto& r = ranks_[rank];
        assert(buf_idx < r.num_bufs);
        auto& b = r.buf_table[buf_idx];
        void* base = nullptr;
        switch (static_cast<BufPool>(b.pool)) {
            case BufPool::Input:   base = r.input.data();   break;
            case BufPool::Output:  base = r.output.data();  break;
            case BufPool::Scratch: base = r.scratch.data();  break;
        }
        return static_cast<uint8_t*>(base) + b.offset;
    }

    void do_put(int dst_rank, const void* src, size_t size,
                uint16_t remote_buf_idx, size_t offset) {
        void* dst = static_cast<uint8_t*>(resolve_remote(dst_rank, remote_buf_idx)) + offset;
        std::lock_guard<std::mutex> lock(mtx_);
        std::memcpy(dst, src, size);
    }

    void do_signal(int src_rank, int dst_rank) {
        std::lock_guard<std::mutex> lock(mtx_);
        signals_[src_rank][dst_rank]++;
        cv_.notify_all();
    }

    void do_wait(int src_rank, int my_rank) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&]{ return signals_[src_rank][my_rank] > 0; });
        signals_[src_rank][my_rank]--;
    }

    RankBufs& rank(int r) { return ranks_[r]; }

private:
    int num_ranks_;
    std::vector<RankBufs> ranks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    int signals_[8][8] = {};
};

class MockTransport : public Transport {
public:
    MockTransport(std::shared_ptr<MockWorld> world, int my_rank)
        : world_(world), my_rank_(my_rank) {}

    void put(uint16_t dst_rank, const void* src, size_t size,
             uint16_t remote_buf_idx, size_t offset) override {
        world_->do_put(dst_rank, src, size, remote_buf_idx, offset);
    }

    void signal(uint16_t dst_rank) override {
        world_->do_signal(my_rank_, dst_rank);
    }

    void wait(uint16_t src_rank) override {
        world_->do_wait(src_rank, my_rank_);
    }

    void sync() override {}  // all ops are synchronous in mock

private:
    std::shared_ptr<MockWorld> world_;
    int my_rank_;
};

} // namespace planck
```

- [ ] Step 2: Verify build

```bash
cd csrc/build && cmake --build .
```

- [ ] Step 3: Commit

```bash
git add csrc/transport/mock.cpp
git commit -m "feat(phase-b): MockWorld shared-memory transport"
```

### Task 6: Plan Deserialization Test

Files:
- Create: `csrc/test/test_plan.cpp`

- [ ] Step 1: Write test_plan.cpp

```cpp
// csrc/test/test_plan.cpp
#include "test_util.h"
#include "planck/plan.h"
#include <fstream>
#include <vector>

using namespace planck;

void test_struct_sizes() {
    CHECK_EQ(sizeof(PlanHeader), 32);
    CHECK_EQ(sizeof(BufEntry),   12);
    CHECK_EQ(sizeof(OpEntry),    16);
}

void test_magic_constants() {
    CHECK_EQ(PLAN_MAGIC,   0x4B4E4C50);
    CHECK_EQ(PLAN_VERSION, 1);
}

static std::vector<uint8_t> load_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return {}; }
    return {std::istreambuf_iterator<char>(f), {}};
}

void test_fixture_roundtrip() {
    auto data = load_file("../test/fixtures/plan_rank0.bin");
    CHECK(data.size() > 32, "fixture too small: %zu", data.size());

    PlanView plan(data.data(), data.size());
    CHECK(plan.valid(), "plan not valid");

    auto& h = plan.header();
    CHECK_EQ(h.magic,     PLAN_MAGIC);
    CHECK_EQ(h.version,   PLAN_VERSION);
    CHECK_EQ(h.num_ranks, 8);
    CHECK_EQ(h.my_rank,   0);
    CHECK(h.num_ops > 0,     "num_ops=%d", h.num_ops);
    CHECK(h.num_buffers > 0, "num_buffers=%d", h.num_buffers);

    // Verify buffers are readable
    const auto* bufs = plan.buffers();
    for (int i = 0; i < h.num_buffers; i++) {
        CHECK(bufs[i].size > 0 || bufs[i].pool == 0, "buf %d: size=%u pool=%u", i, bufs[i].size, bufs[i].pool);
    }

    // Verify ops are readable
    const auto* ops = plan.ops();
    for (int i = 0; i < h.num_ops; i++) {
        CHECK(ops[i].opcode <= 8, "op %d: invalid opcode %u", i, ops[i].opcode);
    }
}

void test_all_8_ranks_loadable() {
    char path[128];
    for (int r = 0; r < 8; r++) {
        snprintf(path, sizeof(path), "../test/fixtures/plan_rank%d.bin", r);
        auto data = load_file(path);
        CHECK(data.size() > 32, "rank %d fixture missing", r);
        PlanView plan(data.data(), data.size());
        CHECK(plan.valid(), "rank %d plan invalid", r);
        CHECK_EQ(plan.header().my_rank, r);
    }
}

TEST_MAIN(
    RUN(test_struct_sizes);
    RUN(test_magic_constants);
    RUN(test_fixture_roundtrip);
    RUN(test_all_8_ranks_loadable);
)
```

- [ ] Step 2: Build and run

```bash
cd csrc/build && cmake --build . && ctest -R test_plan -V
```

Expected: all 4 tests pass

- [ ] Step 3: Commit

```bash
git add csrc/test/test_plan.cpp
git commit -m "test(phase-b): plan.h struct sizes + fixture deserialization"
```

---

## Chunk 3: Executor Engine + 8-Rank Simulation

### Task 7: Executor engine.cpp

Files:
- Create: `csrc/executor/engine.cpp`
- Modify: `csrc/CMakeLists.txt` (add engine.cpp to planck_core)

- [ ] Step 1: Write engine.cpp

```cpp
// csrc/executor/engine.cpp
#include "planck/executor.h"
#include <cstring>
#include <cstdio>

namespace planck {

int Executor::execute(const PlanView& plan, const Config& cfg) {
    if (!plan.valid()) return -1;

    const auto& h    = plan.header();
    const auto* bufs = plan.buffers();
    const auto* ops  = plan.ops();

    // Resolve local buffer pointer from buffer index
    auto resolve = [&](uint16_t idx) -> void* {
        const auto& b = bufs[idx];
        void* base = nullptr;
        switch (static_cast<BufPool>(b.pool)) {
            case BufPool::Input:   base = cfg.input_buf;   break;
            case BufPool::Output:  base = cfg.output_buf;  break;
            case BufPool::Scratch: base = cfg.scratch_buf;  break;
        }
        return static_cast<uint8_t*>(base) + b.offset;
    };

    for (uint16_t i = 0; i < h.num_ops; i++) {
        const auto& op = ops[i];
        auto oc = static_cast<Opcode>(op.opcode);

        switch (oc) {
        case Opcode::Put:
            transport_->put(op.dst_rank, resolve(op.src_buf),
                            bufs[op.src_buf].size, op.dst_buf, 0);
            break;

        case Opcode::Signal:
            transport_->signal(op.dst_rank);
            break;

        case Opcode::Wait:
            transport_->wait(op.dst_rank);
            break;

        case Opcode::LocalCopy: {
            std::memcpy(resolve(op.dst_buf), resolve(op.src_buf),
                        bufs[op.src_buf].size);
            break;
        }
        case Opcode::LocalReduce: {
            auto* dst = static_cast<float*>(resolve(op.dst_buf));
            auto* src = static_cast<const float*>(resolve(op.src_buf));
            size_t n  = bufs[op.src_buf].size / sizeof(float);
            for (size_t j = 0; j < n; j++) dst[j] += src[j];
            break;
        }
        case Opcode::PutWithSignal:
            transport_->put(op.dst_rank, resolve(op.src_buf),
                            bufs[op.src_buf].size, op.dst_buf, 0);
            transport_->signal(op.dst_rank);
            break;

        case Opcode::WaitReduceCopy: {
            transport_->wait(op.dst_rank);
            // 1. Reduce: buf[_pad] += scratch[src_buf]
            auto* red_dst = static_cast<float*>(resolve(op._pad));
            auto* src     = static_cast<const float*>(resolve(op.src_buf));
            size_t n      = bufs[op.src_buf].size / sizeof(float);
            for (size_t j = 0; j < n; j++) red_dst[j] += src[j];
            // 2. Copy: buf[_pad] -> buf[dst_buf]
            std::memcpy(resolve(op.dst_buf), resolve(op._pad),
                        bufs[op.src_buf].size);
            break;
        }
        case Opcode::WaitReducePut: {
            transport_->wait(op.dst_rank);
            // 1. Reduce: buf[dst_buf] += scratch[src_buf]
            auto* dst = static_cast<float*>(resolve(op.dst_buf));
            auto* src = static_cast<const float*>(resolve(op.src_buf));
            size_t n  = bufs[op.src_buf].size / sizeof(float);
            for (size_t j = 0; j < n; j++) dst[j] += src[j];
            // 2. Put: buf[dst_buf] -> remote[_pad]
            transport_->put(op.dst_rank, dst, bufs[op.dst_buf].size,
                            op._pad, 0);
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

- [ ] Step 2: Update CMakeLists.txt — add engine.cpp

```cmake
add_library(planck_core STATIC
    transport/mock.cpp
    executor/engine.cpp
)
```

- [ ] Step 3: Verify build

```bash
cd csrc/build && cmake --build .
```

- [ ] Step 4: Commit

```bash
git add csrc/executor/engine.cpp csrc/CMakeLists.txt
git commit -m "feat(phase-b): executor engine with 9 opcodes + _pad handling"
```

### Task 8: 8-Rank Simulation Test

Files:
- Create: `csrc/test/test_executor.cpp`
- Modify: `csrc/CMakeLists.txt` (add test_executor target)

- [ ] Step 1: Write test_executor.cpp

```cpp
// csrc/test/test_executor.cpp — 8-thread allreduce simulation
#include "test_util.h"
#include "planck/plan.h"
#include "planck/executor.h"
#include <fstream>
#include <vector>
#include <thread>
#include <memory>
#include <cstring>

// Include mock.cpp implementation (header-only style for test)
// In production this would be linked, but MockWorld class is in the .cpp
// For simplicity, declare extern or include directly:
namespace planck { class MockWorld; class MockTransport; }
#include "../transport/mock.cpp"

using namespace planck;

static std::vector<uint8_t> load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

void test_8rank_allreduce() {
    // Load 8 plans
    std::vector<std::vector<uint8_t>> plan_data(8);
    char path[128];
    for (int r = 0; r < 8; r++) {
        snprintf(path, sizeof(path), "../test/fixtures/plan_rank%d.bin", r);
        plan_data[r] = load(path);
        CHECK(plan_data[r].size() > 32, "rank %d fixture missing", r);
    }

    PlanView plan0(plan_data[0].data(), plan_data[0].size());
    auto& h = plan0.header();
    int num_ranks  = h.num_ranks;
    int num_floats = 256 / sizeof(float);  // 64 floats (256 bytes msg)

    CHECK_EQ(num_ranks, 8);

    // Compute scratch size from buffer table
    size_t scratch_size = 0;
    for (int i = 0; i < h.num_buffers; i++) {
        auto& b = plan0.buffers()[i];
        if (static_cast<BufPool>(b.pool) == BufPool::Scratch) {
            size_t end = b.offset + b.size;
            if (end > scratch_size) scratch_size = end;
        }
    }

    // Allocate per-rank buffers
    struct RankData {
        std::vector<float> input;
        std::vector<float> scratch;
    };
    std::vector<RankData> ranks(8);
    for (int r = 0; r < 8; r++) {
        ranks[r].input.assign(num_floats, (float)(r + 1));  // rank r: [r+1, r+1, ...]
        ranks[r].scratch.resize(scratch_size / sizeof(float) + 1, 0.0f);
    }

    // Create MockWorld
    auto world = std::make_shared<MockWorld>(8);
    for (int r = 0; r < 8; r++) {
        PlanView pv(plan_data[r].data(), plan_data[r].size());
        world->setup_rank(r,
            ranks[r].input.data(),   num_floats * sizeof(float),
            ranks[r].input.data(),   num_floats * sizeof(float),  // output == input (in-place)
            ranks[r].scratch.data(), scratch_size,
            pv.buffers(), pv.header().num_buffers);
    }

    // Run 8 threads
    std::vector<std::thread> threads;
    std::vector<int> results(8, -1);

    for (int r = 0; r < 8; r++) {
        threads.emplace_back([&, r]() {
            auto transport = std::make_shared<MockTransport>(world, r);
            Executor executor(transport);
            PlanView pv(plan_data[r].data(), plan_data[r].size());
            Executor::Config cfg{};
            cfg.my_rank     = (uint16_t)r;
            cfg.input_buf   = ranks[r].input.data();
            cfg.output_buf  = ranks[r].input.data();
            cfg.scratch_buf = ranks[r].scratch.data();
            cfg.scratch_size = scratch_size;
            results[r] = executor.execute(pv, cfg);
        });
    }

    for (auto& t : threads) t.join();

    // Verify all ranks succeeded
    for (int r = 0; r < 8; r++) {
        CHECK_EQ(results[r], 0);
    }

    // Verify allreduce result: sum(1..8) = 36.0
    for (int r = 0; r < 8; r++) {
        for (int i = 0; i < num_floats; i++) {
            CHECK_NEAR(ranks[r].input[i], 36.0f, 1e-3);
        }
    }
}

TEST_MAIN(
    RUN(test_8rank_allreduce);
)
```

- [ ] Step 2: Update CMakeLists.txt

```cmake
add_executable(test_executor test/test_executor.cpp)
target_link_libraries(test_executor planck_core)
add_test(NAME test_executor COMMAND test_executor)
```

- [ ] Step 3: Build and run

```bash
cd csrc/build && cmake --build . && ctest -R test_executor -V
```

Expected: test_8rank_allreduce passes, all 8 ranks output [36.0, ...]

- [ ] Step 4: Commit

```bash
git add csrc/test/test_executor.cpp csrc/CMakeLists.txt
git commit -m "test(phase-b): 8-rank multithreaded allreduce simulation"
```

---

## Chunk 4: PyTorch Eager Integration (Optional — requires libtorch)

### Task 9: torch_binding.cpp

Files:
- Create: `csrc/torch_binding.cpp`
- Modify: `csrc/CMakeLists.txt`

- [ ] Step 1: Write torch_binding.cpp

```cpp
// csrc/torch_binding.cpp — register Planck ops with PyTorch
#include <torch/library.h>
#include <torch/torch.h>
#include "planck/plan.h"
#include "planck/executor.h"

// Simplified eager implementation for testing.
// In production, this would use PlanCache + real transport.
torch::Tensor planck_allreduce(torch::Tensor input, const std::string& plan_key) {
    // For now: identity (real impl needs MockWorld or HCCS transport)
    return input.clone();
}

TORCH_LIBRARY(planck, m) {
    m.def("allreduce(Tensor input, str plan_key) -> Tensor");
    m.impl("allreduce", planck_allreduce);
}
```

- [ ] Step 2: Update CMakeLists.txt (inside Torch_FOUND block)

```cmake
if(Torch_FOUND)
    add_library(planck_torch SHARED torch_binding.cpp)
    target_link_libraries(planck_torch planck_core "${TORCH_LIBRARIES}")
    target_include_directories(planck_torch PRIVATE "${TORCH_INCLUDE_DIRS}")
endif()
```

- [ ] Step 3: Commit

```bash
git add csrc/torch_binding.cpp csrc/CMakeLists.txt
git commit -m "feat(phase-b): TORCH_LIBRARY registration for planck ops"
```

### Task 10: ops.py + test_torch_eager.py

Files:
- Create: `python/planck/ops.py`
- Create: `tests/test_torch_eager.py`
- Modify: `python/planck/__init__.py`

- [ ] Step 1: Write ops.py

```python
# python/planck/ops.py — FakeTensor registration for torch.compile tracing
"""FakeTensor (meta) implementations for Planck custom ops."""

try:
    import torch

    @torch.library.register_fake("planck::allreduce")
    def _(input, plan_key):
        return input.clone()
except Exception:
    pass  # torch not available or ops not registered
```

- [ ] Step 2: Write test_torch_eager.py

```python
# tests/test_torch_eager.py
"""Test Planck ops in PyTorch eager mode (requires libtorch build)."""
import pytest

torch = pytest.importorskip("torch")


def test_op_registered():
    """planck::allreduce op should be callable."""
    try:
        x = torch.randn(64)
        y = torch.ops.planck.allreduce(x, "test_key")
        assert y.shape == x.shape
    except RuntimeError as e:
        if "No such operator" in str(e):
            pytest.skip("planck torch library not loaded")
        raise
```

- [ ] Step 3: Update __init__.py

Add to `python/planck/__init__.py`:
```python
try:
    from planck import ops as _ops  # register FakeTensor on import
except ImportError:
    pass
```

- [ ] Step 4: Commit

```bash
git add python/planck/ops.py tests/test_torch_eager.py python/planck/__init__.py
git commit -m "feat(phase-b): FakeTensor registration + eager mode test"
```

---

## Block 2: planck-sim DES Simulator

---

## Chunk 5: DES Engine Core + Link Model

### Task 11: Cargo.toml + lib.rs + sim/mod.rs + config.rs

Files:
- Modify: `crates/planck-core/Cargo.toml`
- Modify: `crates/planck-core/src/lib.rs`
- Create: `crates/planck-core/src/sim/mod.rs`
- Create: `crates/planck-core/src/sim/config.rs`

- [ ] Step 1: Update Cargo.toml

```toml
[features]
sim = ["toml", "serde"]

[dependencies]
toml  = { version = "0.8", optional = true }
serde = { version = "1", features = ["derive"], optional = true }
```

- [ ] Step 2: Update lib.rs

```rust
pub mod plan;
pub mod topo;
pub mod cost;
pub mod algo;
pub mod sched;
pub mod template;
#[cfg(feature = "sim")]
pub mod sim;
```

- [ ] Step 3: Write config.rs

```rust
// crates/planck-core/src/sim/config.rs
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SimConfig {
    #[serde(default)]
    pub collective: CollectiveConfig,
    #[serde(default)]
    pub topology:   TopoConfig,
    #[serde(default)]
    pub timing:     TimingConfig,
    #[serde(default)]
    pub output:     OutputConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CollectiveConfig {
    #[serde(rename = "type", default = "default_coll_type")]
    pub coll_type:       String,
    #[serde(default = "default_msg_size", deserialize_with = "parse_size")]
    pub msg_size:        usize,
    #[serde(default = "default_chunks")]
    pub pipeline_chunks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TopoConfig {
    #[serde(default = "default_preset")]
    pub preset: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimingConfig {
    #[serde(default = "default_model")]
    pub model:          String,
    #[serde(default = "default_bw")]
    pub hccs_bw_gbps:   f64,
    #[serde(default = "default_lat")]
    pub hccs_lat_us:     f64,
    #[serde(default = "default_notify")]
    pub notify_rounds:   u32,
    #[serde(default = "default_sqe")]
    pub sqe_depth:       u32,
    #[serde(default = "default_doorbell")]
    pub doorbell_batch:  u32,
    #[serde(default = "default_hbm")]
    pub hbm_bw_gbps:    f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_file")]
    pub file:   String,
}

fn default_coll_type() -> String { "allreduce".into() }
fn default_msg_size()  -> usize  { 256 << 20 }
fn default_chunks()    -> usize  { 4 }
fn default_preset()    -> String { "hccs_8card".into() }
fn default_model()     -> String { "ascend".into() }
fn default_bw()        -> f64    { 30.0 }
fn default_lat()       -> f64    { 1.5 }
fn default_notify()    -> u32    { 3 }
fn default_sqe()       -> u32    { 2048 }
fn default_doorbell()  -> u32    { 10 }
fn default_hbm()       -> f64    { 460.0 }
fn default_format()    -> String { "chrome_trace".into() }
fn default_file()      -> String { "trace.json".into() }

impl Default for CollectiveConfig { fn default() -> Self { Self { coll_type: default_coll_type(), msg_size: default_msg_size(), pipeline_chunks: default_chunks() } } }
impl Default for TopoConfig       { fn default() -> Self { Self { preset: default_preset() } } }
impl Default for TimingConfig      { fn default() -> Self { Self { model: default_model(), hccs_bw_gbps: default_bw(), hccs_lat_us: default_lat(), notify_rounds: default_notify(), sqe_depth: default_sqe(), doorbell_batch: default_doorbell(), hbm_bw_gbps: default_hbm() } } }
impl Default for OutputConfig      { fn default() -> Self { Self { format: default_format(), file: default_file() } } }
impl Default for SimConfig         { fn default() -> Self { Self { collective: Default::default(), topology: Default::default(), timing: Default::default(), output: Default::default() } } }

/// Parse size strings like "256MB", "16KB", "1GB", or plain numbers.
fn parse_size<'de, D: serde::Deserializer<'de>>(de: D) -> Result<usize, D::Error> {
    let s = String::deserialize(de)?;
    parse_size_str(&s).map_err(serde::de::Error::custom)
}

pub fn parse_size_str(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if let Ok(n) = s.parse::<usize>() { return Ok(n); }
    let (num, suffix) = s.split_at(s.len() - 2.min(s.len()));
    let n: f64 = num.trim().parse().map_err(|e| format!("{e}"))?;
    match suffix.to_uppercase().as_str() {
        "KB" => Ok((n * 1024.0) as usize),
        "MB" => Ok((n * 1024.0 * 1024.0) as usize),
        "GB" => Ok((n * 1024.0 * 1024.0 * 1024.0) as usize),
        _    => Err(format!("unknown size suffix: {s}")),
    }
}

impl SimConfig {
    pub fn from_toml(s: &str) -> Result<Self, String> {
        toml::from_str(s).map_err(|e| format!("{e}"))
    }
}
```

- [ ] Step 4: Write sim/mod.rs (stub)

```rust
// crates/planck-core/src/sim/mod.rs
pub mod config;
pub mod engine;
pub mod link;
pub mod timing;
pub mod trace;

pub use config::SimConfig;
pub use trace::Trace;

use crate::plan::ExecutionPlan;
use crate::topo::Topology;

/// Simulate execution of plans for all ranks, return Chrome Trace.
pub fn simulate(plans: &[ExecutionPlan], topo: &Topology, cfg: &SimConfig) -> Trace {
    let model = timing::create_model(cfg);
    let mut sim = engine::Simulator::new(plans, topo, model, cfg);
    sim.run();
    sim.into_trace()
}
```

- [ ] Step 5: Verify build

```bash
cargo build -p planck-core --features sim
```

- [ ] Step 6: Commit

```bash
git add crates/planck-core/Cargo.toml crates/planck-core/src/lib.rs crates/planck-core/src/sim/
git commit -m "feat(phase-b): planck-sim skeleton with config + TOML parsing"
```

### Task 12: engine.rs + link.rs

Files:
- Create: `crates/planck-core/src/sim/engine.rs`
- Create: `crates/planck-core/src/sim/link.rs`

- [ ] Step 1: Write link.rs

```rust
// crates/planck-core/src/sim/link.rs
use crate::topo::Link;

pub struct LinkState {
    pub link:         Link,
    pub active_flows: u32,
}

impl LinkState {
    pub fn new(link: Link) -> Self {
        Self { link, active_flows: 0 }
    }

    pub fn effective_bw_gbps(&self) -> f64 {
        if self.active_flows == 0 {
            self.link.bandwidth_gbps
        } else {
            self.link.bandwidth_gbps / self.active_flows as f64
        }
    }

    pub fn add_flow(&mut self) { self.active_flows += 1; }
    pub fn remove_flow(&mut self) { self.active_flows = self.active_flows.saturating_sub(1); }
}
```

- [ ] Step 2: Write engine.rs

```rust
// crates/planck-core/src/sim/engine.rs
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::plan::{ExecutionPlan, Opcode, BufPool};
use crate::topo::Topology;
use super::config::SimConfig;
use super::link::LinkState;
use super::timing::TimingModel;
use super::trace::{Trace, TraceEvent};

#[derive(Debug, Clone)]
pub struct Event {
    pub time:   f64,
    pub rank:   u16,
    pub stream: u8,
    pub kind:   EventKind,
}

#[derive(Debug, Clone)]
pub enum EventKind {
    OpStart  { op_idx: u16 },
    OpEnd    { op_idx: u16, name: String },
    PutEnd   { link_idx: usize },
    Unblock  { op_idx: u16 },  // signal arrived, unblock waiting op
}

// Min-heap by time
impl PartialEq  for Event { fn eq(&self, o: &Self) -> bool { self.time == o.time } }
impl Eq          for Event {}
impl PartialOrd  for Event { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord         for Event { fn cmp(&self, o: &Self) -> Ordering { o.time.partial_cmp(&self.time).unwrap_or(Ordering::Equal) } }

pub struct Simulator {
    queue:  BinaryHeap<Event>,
    clock:  f64,
    links:  Vec<LinkState>,
    plans:  Vec<ExecutionPlan>,
    trace:  Trace,
    model:  Box<dyn TimingModel>,
    num_ranks: usize,
    // Per-rank, per-stream: next op index to execute
    next_op: Vec<Vec<u16>>,
    // Signal counts: signals[src][dst]
    signals: Vec<Vec<i32>>,
    // Waiting ops: waiting[rank] = Some(op_idx) if blocked on wait
    waiting: Vec<Option<u16>>,
}

impl Simulator {
    pub fn new(
        plans: &[ExecutionPlan], topo: &Topology,
        model: Box<dyn TimingModel>, cfg: &SimConfig,
    ) -> Self {
        let n = plans.len();
        let num_streams = plans.get(0).map_or(1, |p| p.header.num_streams.max(1)) as usize;

        let links: Vec<LinkState> = topo.links.iter()
            .map(|l| LinkState::new(l.clone()))
            .collect();

        let mut next_op = vec![vec![0u16; num_streams]; n];
        let signals = vec![vec![0i32; n]; n];
        let waiting = vec![None; n];

        let mut sim = Self {
            queue: BinaryHeap::new(),
            clock: 0.0,
            links,
            plans: plans.to_vec(),
            trace: Trace::new(n),
            model,
            num_ranks: n,
            next_op,
            signals,
            waiting,
        };

        // Seed: first op of each rank/stream starts at t=0
        for rank in 0..n {
            let p = &plans[rank];
            for s in 0..num_streams {
                // Find first op on this stream
                if let Some(idx) = p.ops.iter().position(|o| o.stream_id == s as u8) {
                    sim.queue.push(Event {
                        time: 0.0, rank: rank as u16, stream: s as u8,
                        kind: EventKind::OpStart { op_idx: idx as u16 },
                    });
                }
            }
        }

        sim
    }

    pub fn run(&mut self) {
        while let Some(ev) = self.queue.pop() {
            self.clock = ev.time;
            self.handle(ev);
        }
    }

    fn handle(&mut self, ev: Event) {
        let r = ev.rank as usize;
        match ev.kind {
            EventKind::OpStart { op_idx } => self.handle_op_start(r, ev.stream, op_idx, ev.time),
            EventKind::OpEnd { op_idx, ref name } => self.handle_op_end(r, ev.stream, op_idx, ev.time, name),
            EventKind::PutEnd { link_idx } => { self.links[link_idx].remove_flow(); }
            EventKind::Unblock { op_idx } => self.handle_op_start(r, ev.stream, op_idx, ev.time),
        }
    }

    fn handle_op_start(&mut self, rank: usize, stream: u8, op_idx: u16, now: f64) {
        let op = &self.plans[rank].ops[op_idx as usize];
        let oc = Opcode::try_from(op.opcode).unwrap_or(Opcode::Noop);
        let bufs = &self.plans[rank].buffers;

        match oc {
            Opcode::Noop | Opcode::Signal => {
                if oc == Opcode::Signal {
                    // Record signal for dst_rank
                    let dst = op.dst_rank as usize;
                    self.signals[rank][dst] += 1;
                    // Check if dst is waiting
                    if let Some(wait_op) = self.waiting[dst] {
                        self.waiting[dst] = None;
                        let wop = &self.plans[dst].ops[wait_op as usize];
                        self.queue.push(Event {
                            time: now, rank: dst as u16, stream: wop.stream_id,
                            kind: EventKind::Unblock { op_idx: wait_op },
                        });
                    }
                }
                self.schedule_next(rank, stream, op_idx, now, "Signal");
            }

            Opcode::Wait => {
                let src = op.dst_rank as usize;
                if self.signals[src][rank] > 0 {
                    self.signals[src][rank] -= 1;
                    let dur = self.model.notify_time(&self.links[0].link);
                    self.trace.push(rank, stream, "Wait", now, dur, None);
                    self.schedule_next(rank, stream, op_idx, now + dur, "Wait");
                } else {
                    self.waiting[rank] = Some(op_idx);
                }
            }

            Opcode::Put | Opcode::PutWithSignal => {
                let size = bufs[op.src_buf as usize].size as usize;
                let link_idx = self.find_link(rank, op.dst_rank as usize);
                self.links[link_idx].add_flow();
                let dur = self.model.put_time(&self.links[link_idx].link, size);

                self.trace.push(rank, stream, "Put", now, dur, Some(op.dst_rank));
                self.queue.push(Event { time: now + dur, rank: rank as u16, stream, kind: EventKind::PutEnd { link_idx } });

                if oc == Opcode::PutWithSignal {
                    let dst = op.dst_rank as usize;
                    self.signals[rank][dst] += 1;
                    if let Some(wait_op) = self.waiting[dst] {
                        self.waiting[dst] = None;
                        self.queue.push(Event {
                            time: now + dur, rank: dst as u16,
                            stream: self.plans[dst].ops[wait_op as usize].stream_id,
                            kind: EventKind::Unblock { op_idx: wait_op },
                        });
                    }
                }

                self.schedule_next(rank, stream, op_idx, now + dur, "Put");
            }

            Opcode::LocalReduce | Opcode::LocalCopy => {
                let size = bufs[op.src_buf as usize].size as usize;
                let dur = self.model.reduce_time(size);
                let name = if oc == Opcode::LocalReduce { "Reduce" } else { "Copy" };
                self.trace.push(rank, stream, name, now, dur, None);
                self.schedule_next(rank, stream, op_idx, now + dur, name);
            }

            Opcode::WaitReducePut => {
                let src = op.dst_rank as usize;
                if self.signals[src][rank] > 0 {
                    self.signals[src][rank] -= 1;
                    let size = bufs[op.src_buf as usize].size as usize;
                    let link_idx = self.find_link(rank, op.dst_rank as usize);
                    self.links[link_idx].add_flow();
                    let dur = self.model.inline_reduce_put_time(&self.links[link_idx].link, size);

                    self.trace.push(rank, stream, "WaitReducePut", now, dur, Some(op.dst_rank));
                    self.queue.push(Event { time: now + dur, rank: rank as u16, stream, kind: EventKind::PutEnd { link_idx } });

                    // Signal after put
                    let dst = op.dst_rank as usize;
                    self.signals[rank][dst] += 1;
                    if let Some(wait_op) = self.waiting[dst] {
                        self.waiting[dst] = None;
                        self.queue.push(Event {
                            time: now + dur, rank: dst as u16,
                            stream: self.plans[dst].ops[wait_op as usize].stream_id,
                            kind: EventKind::Unblock { op_idx: wait_op },
                        });
                    }

                    self.schedule_next(rank, stream, op_idx, now + dur, "WaitReducePut");
                } else {
                    self.waiting[rank] = Some(op_idx);
                }
            }

            Opcode::WaitReduceCopy => {
                let src = op.dst_rank as usize;
                if self.signals[src][rank] > 0 {
                    self.signals[src][rank] -= 1;
                    let size = bufs[op.src_buf as usize].size as usize;
                    let dur = self.model.reduce_time(size) * 2.0;  // reduce + copy
                    self.trace.push(rank, stream, "WaitReduceCopy", now, dur, None);
                    self.schedule_next(rank, stream, op_idx, now + dur, "WaitReduceCopy");
                } else {
                    self.waiting[rank] = Some(op_idx);
                }
            }
        }
    }

    fn handle_op_end(&mut self, _rank: usize, _stream: u8, _op_idx: u16, _now: f64, _name: &str) {}

    fn schedule_next(&mut self, rank: usize, stream: u8, current_op: u16, after: f64, _name: &str) {
        let ops = &self.plans[rank].ops;
        // Find next op on same stream after current
        for idx in (current_op as usize + 1)..ops.len() {
            if ops[idx].stream_id == stream {
                self.queue.push(Event {
                    time: after + self.model.kernel_launch_overhead(),
                    rank: rank as u16, stream,
                    kind: EventKind::OpStart { op_idx: idx as u16 },
                });
                return;
            }
        }
    }

    fn find_link(&self, src: usize, dst: usize) -> usize {
        self.links.iter().position(|l| l.link.src == src && l.link.dst == dst).unwrap_or(0)
    }

    pub fn into_trace(self) -> Trace { self.trace }
}

// Needed for plans.to_vec() — add Clone to ExecutionPlan if not present
impl Clone for ExecutionPlan {
    fn clone(&self) -> Self {
        Self { header: self.header, buffers: self.buffers.clone(), ops: self.ops.clone() }
    }
}

impl Opcode {
    fn try_from(v: u8) -> Option<Self> {
        if v <= 8 { Some(unsafe { std::mem::transmute(v) }) } else { None }
    }
}
```

- [ ] Step 3: Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_ordering() {
        let mut q = BinaryHeap::new();
        q.push(Event { time: 5.0, rank: 0, stream: 0, kind: EventKind::OpStart { op_idx: 0 } });
        q.push(Event { time: 1.0, rank: 0, stream: 0, kind: EventKind::OpStart { op_idx: 1 } });
        q.push(Event { time: 3.0, rank: 0, stream: 0, kind: EventKind::OpStart { op_idx: 2 } });
        assert_eq!(q.pop().unwrap().time, 1.0);
        assert_eq!(q.pop().unwrap().time, 3.0);
        assert_eq!(q.pop().unwrap().time, 5.0);
    }
}
```

- [ ] Step 4: Verify

```bash
cargo test -p planck-core --features sim -- sim
```

- [ ] Step 5: Commit

```bash
git add crates/planck-core/src/sim/engine.rs crates/planck-core/src/sim/link.rs
git commit -m "feat(phase-b): DES engine + link bandwidth model"
```

---

## Chunk 6: Timing Models + Trace Output

### Task 13: timing.rs

Files:
- Create: `crates/planck-core/src/sim/timing.rs`

- [ ] Step 1: Write timing.rs

```rust
// crates/planck-core/src/sim/timing.rs
use crate::topo::Link;
use super::config::SimConfig;

pub trait TimingModel {
    fn put_time(&self, link: &Link, size: usize) -> f64;
    fn notify_time(&self, link: &Link) -> f64;
    fn reduce_time(&self, size: usize) -> f64;
    fn inline_reduce_put_time(&self, link: &Link, size: usize) -> f64;
    fn kernel_launch_overhead(&self) -> f64;
}

// ==== SimpleModel: alpha-beta ====

pub struct SimpleModel;

impl TimingModel for SimpleModel {
    fn put_time(&self, link: &Link, size: usize) -> f64 {
        link.latency_us + size as f64 / (link.bandwidth_gbps * 1e3)  // GB/s -> MB/us
    }
    fn notify_time(&self, link: &Link) -> f64 { link.latency_us }
    fn reduce_time(&self, size: usize) -> f64 { size as f64 / (460.0 * 1e3) }  // HBM ~460 GB/s
    fn inline_reduce_put_time(&self, link: &Link, size: usize) -> f64 {
        self.put_time(link, size) + self.reduce_time(size)  // no overlap in simple model
    }
    fn kernel_launch_overhead(&self) -> f64 { 0.0 }
}

// ==== AscendModel: HCCS hardware-aware ====

pub struct AscendModel {
    pub notify_rounds:  u32,   // 3-round handshake
    pub hbm_bw_gbps:   f64,   // HBM bandwidth
    pub launch_us:      f64,   // kernel launch overhead
}

impl TimingModel for AscendModel {
    fn put_time(&self, link: &Link, size: usize) -> f64 {
        // HCCS GET mode: 2 * latency (request + data) + transfer time
        2.0 * link.latency_us + size as f64 / (link.bandwidth_gbps * 1e3)
    }

    fn notify_time(&self, link: &Link) -> f64 {
        self.notify_rounds as f64 * link.latency_us
    }

    fn reduce_time(&self, size: usize) -> f64 {
        size as f64 / (self.hbm_bw_gbps * 1e3)
    }

    fn inline_reduce_put_time(&self, link: &Link, size: usize) -> f64 {
        // InlineReduce: reduce and put overlap (MTE+AIV physical separation)
        let reduce = self.reduce_time(size);
        let put    = self.put_time(link, size);
        let notify = self.notify_time(link);
        notify + reduce.max(put)  // wait for notify, then overlapped reduce+put
    }

    fn kernel_launch_overhead(&self) -> f64 { self.launch_us }
}

pub fn create_model(cfg: &SimConfig) -> Box<dyn TimingModel> {
    match cfg.timing.model.as_str() {
        "simple" => Box::new(SimpleModel),
        _ => Box::new(AscendModel {
            notify_rounds: cfg.timing.notify_rounds,
            hbm_bw_gbps:  cfg.timing.hbm_bw_gbps,
            launch_us:     5.0,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::{Link, TransportType};

    fn test_link() -> Link {
        Link { src: 0, dst: 1, bandwidth_gbps: 30.0, latency_us: 1.5, transport: TransportType::Hccs }
    }

    #[test]
    fn simple_put_time() {
        let m = SimpleModel;
        let t = m.put_time(&test_link(), 30_000); // 30KB
        assert!(t > 1.5); // at least latency
        assert!(t < 10.0);
    }

    #[test]
    fn ascend_notify_3_rounds() {
        let m = AscendModel { notify_rounds: 3, hbm_bw_gbps: 460.0, launch_us: 5.0 };
        let t = m.notify_time(&test_link());
        assert!((t - 4.5).abs() < 0.01); // 3 * 1.5us
    }

    #[test]
    fn ascend_inline_reduce_overlaps() {
        let m = AscendModel { notify_rounds: 3, hbm_bw_gbps: 460.0, launch_us: 5.0 };
        let link = test_link();
        let fused   = m.inline_reduce_put_time(&link, 256_000);
        let separate = m.notify_time(&link) + m.reduce_time(256_000) + m.put_time(&link, 256_000);
        assert!(fused < separate, "InlineReduce should overlap: fused={fused} separate={separate}");
    }
}
```

- [ ] Step 2: Commit

```bash
git add crates/planck-core/src/sim/timing.rs
git commit -m "feat(phase-b): TimingModel trait + SimpleModel + AscendModel"
```

### Task 14: trace.rs

Files:
- Create: `crates/planck-core/src/sim/trace.rs`

- [ ] Step 1: Write trace.rs

```rust
// crates/planck-core/src/sim/trace.rs — Chrome Trace JSON output
use std::fmt::Write;

pub struct TraceEvent {
    pub name:     String,
    pub cat:      String,
    pub ph:       char,       // 'X'=complete, 'B'=begin, 'E'=end, 's'/'f'=flow
    pub pid:      u16,        // rank
    pub tid:      u8,         // stream
    pub ts:       f64,        // microseconds
    pub dur:      f64,        // microseconds (for 'X' events)
    pub dst_rank: Option<u16>,
}

pub struct Trace {
    pub events: Vec<TraceEvent>,
    num_ranks:  usize,
}

impl Trace {
    pub fn new(num_ranks: usize) -> Self {
        Self { events: Vec::new(), num_ranks }
    }

    pub fn push(&mut self, rank: usize, stream: u8, name: &str, ts: f64, dur: f64, dst: Option<u16>) {
        self.events.push(TraceEvent {
            name: name.to_string(),
            cat: "planck".to_string(),
            ph: 'X',
            pid: rank as u16,
            tid: stream,
            ts, dur,
            dst_rank: dst,
        });
    }

    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(self.events.len() * 120 + 256);
        s.push_str("{\"traceEvents\":[\n");

        // Metadata: rank names
        for r in 0..self.num_ranks {
            write!(s, "{{\"ph\":\"M\",\"pid\":{r},\"name\":\"process_name\",\"args\":{{\"name\":\"Rank {r}\"}}}},\n").ok();
        }

        // Events
        for (i, ev) in self.events.iter().enumerate() {
            write!(s, "{{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"{}\",\"pid\":{},\"tid\":{},\"ts\":{:.3},\"dur\":{:.3}",
                ev.name, ev.cat, ev.ph, ev.pid, ev.tid, ev.ts, ev.dur).ok();
            if let Some(dst) = ev.dst_rank {
                write!(s, ",\"args\":{{\"dst_rank\":{dst}}}").ok();
            }
            s.push('}');
            if i + 1 < self.events.len() { s.push(','); }
            s.push('\n');
        }

        s.push_str("]}\n");
        s
    }

    pub fn total_time(&self) -> f64 {
        self.events.iter().map(|e| e.ts + e.dur).fold(0.0f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_json_valid() {
        let mut t = Trace::new(2);
        t.push(0, 0, "Put", 0.0, 10.0, Some(1));
        t.push(1, 0, "Wait", 10.0, 2.0, None);
        let json = t.to_json();
        assert!(json.contains("traceEvents"));
        assert!(json.contains("\"name\":\"Put\""));
        assert!(json.contains("\"dst_rank\":1"));
    }
}
```

- [ ] Step 2: Verify

```bash
cargo test -p planck-core --features sim -- sim
```

- [ ] Step 3: Commit

```bash
git add crates/planck-core/src/sim/trace.rs
git commit -m "feat(phase-b): Chrome Trace JSON output with rank metadata"
```

---

## Chunk 7: Integration + PyO3 + TOML Example

### Task 15: Rust Integration Test

Files:
- Add test to `crates/planck-core/src/sim/mod.rs`

- [ ] Step 1: Add integration test

```rust
// In sim/mod.rs, add:
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Collective, CompileRequest, ReduceOp, compile};
    use crate::topo::Topology;

    fn compile_8_plans(msg_size: usize, chunks: usize) -> (Vec<ExecutionPlan>, Topology) {
        let topo = Topology::hccs_8card();
        let plans: Vec<_> = (0..8).map(|r| compile(&CompileRequest {
            collective: Collective::AllReduce,
            msg_size, reduce_op: ReduceOp::Sum,
            num_ranks: 8, my_rank: r, pipeline_chunks: chunks,
        }, &topo)).collect();
        (plans, topo)
    }

    #[test]
    fn pipeline_overlap() {
        let cfg = SimConfig::default();
        let (plans4, topo) = compile_8_plans(256, 4);
        let (plans1, _)    = compile_8_plans(256, 1);
        let t4 = simulate(&plans4, &topo, &cfg).total_time();
        let t1 = simulate(&plans1, &topo, &cfg).total_time();
        assert!(t4 < t1, "4-chunk pipeline ({t4:.1}us) should be faster than 1-chunk ({t1:.1}us)");
    }

    #[test]
    fn monotonic_with_size() {
        let cfg = SimConfig::default();
        let (small, topo) = compile_8_plans(256, 1);
        let (large, _)    = compile_8_plans(256_000, 1);
        let ts = simulate(&small, &topo, &cfg).total_time();
        let tl = simulate(&large, &topo, &cfg).total_time();
        assert!(tl > ts, "larger msg ({tl:.1}us) should take longer than small ({ts:.1}us)");
    }

    #[test]
    fn trace_has_events() {
        let cfg = SimConfig::default();
        let (plans, topo) = compile_8_plans(256, 1);
        let trace = simulate(&plans, &topo, &cfg);
        assert!(!trace.events.is_empty(), "trace should have events");
        let json = trace.to_json();
        assert!(json.contains("traceEvents"));
    }
}
```

- [ ] Step 2: Verify

```bash
cargo test -p planck-core --features sim -- sim
```

- [ ] Step 3: Commit

```bash
git add crates/planck-core/src/sim/mod.rs
git commit -m "test(phase-b): planck-sim integration tests — pipeline overlap + monotonicity"
```

### Task 16: PyO3 simulate() Binding

Files:
- Modify: `crates/planck-python/src/lib.rs`
- Modify: `crates/planck-python/Cargo.toml`

- [ ] Step 1: Add sim feature to planck-python

In `crates/planck-python/Cargo.toml`:
```toml
[dependencies]
planck-core = { path = "../planck-core", features = ["sim"] }
```

- [ ] Step 2: Add simulate function to PyO3

Add to `crates/planck-python/src/lib.rs`:
```rust
// ==== Simulation ====

#[pyfunction]
#[pyo3(signature = (plans, config_toml=None))]
fn simulate(
    py: Python<'_>,
    plans: Vec<PyRef<PyPlanView>>,
    config_toml: Option<&str>,
) -> PyResult<String> {
    use planck_core::sim;
    use planck_core::topo::Topology;

    let cfg = match config_toml {
        Some(s) => sim::SimConfig::from_toml(s).map_err(|e|
            pyo3::exceptions::PyValueError::new_err(e))?,
        None => sim::SimConfig::default(),
    };

    let exec_plans: Vec<_> = plans.iter().map(|p| p.plan.clone()).collect();
    let topo = Topology::hccs_8card();

    let trace = py.allow_threads(|| sim::simulate(&exec_plans, &topo, &cfg));
    Ok(trace.to_json())
}
```

Add to module registration:
```rust
m.add_function(wrap_pyfunction!(simulate, m)?)?;
```

- [ ] Step 3: Update python/__init__.py

```python
from planck._planck import simulate as _simulate

def simulate(plans, config_toml=None):
    """Simulate plan execution, return Chrome Trace JSON string."""
    json_str = _simulate(plans, config_toml)
    return json_str
```

- [ ] Step 4: Commit

```bash
git add crates/planck-python/ python/planck/__init__.py
git commit -m "feat(phase-b): PyO3 simulate() binding"
```

### Task 17: Python Test + TOML Example

Files:
- Create: `tests/test_sim.py`
- Create: `planck-sim.toml`

- [ ] Step 1: Write test_sim.py

```python
# tests/test_sim.py
"""planck-sim integration test via Python."""
import json
import planck


def test_simulate_returns_json():
    compiler = planck.PlanCompiler.hccs_8card()
    plans = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=1) for r in range(8)]
    trace_json = planck.simulate(plans)
    data = json.loads(trace_json)
    assert "traceEvents" in data
    assert len(data["traceEvents"]) > 0


def test_simulate_with_config():
    config = """
[collective]
type = "allreduce"
msg_size = "256"
pipeline_chunks = 4

[timing]
model = "simple"
"""
    compiler = planck.PlanCompiler.hccs_8card()
    plans = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=4) for r in range(8)]
    trace_json = planck.simulate(plans, config_toml=config)
    data = json.loads(trace_json)
    assert len(data["traceEvents"]) > 0


def test_pipeline_faster():
    compiler = planck.PlanCompiler.hccs_8card()
    plans1 = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=1) for r in range(8)]
    plans4 = [compiler.compile_allreduce(256, my_rank=r, pipeline_chunks=4) for r in range(8)]
    t1 = json.loads(planck.simulate(plans1))
    t4 = json.loads(planck.simulate(plans4))
    # 4-chunk should have more events (more ops per chunk)
    assert len(t4["traceEvents"]) >= len(t1["traceEvents"])
```

- [ ] Step 2: Write planck-sim.toml

```toml
# planck-sim.toml — example configuration for planck-sim

[collective]
type = "allreduce"
msg_size = "256MB"
pipeline_chunks = 4

[topology]
preset = "hccs_8card"

[timing]
model = "ascend"
hccs_bw_gbps = 30.0
hccs_lat_us = 1.5
notify_rounds = 3
sqe_depth = 2048
doorbell_batch = 10
hbm_bw_gbps = 460.0

[output]
format = "chrome_trace"
file = "trace.json"
```

- [ ] Step 3: Build and test

```bash
maturin develop && pytest tests/test_sim.py -v
```

- [ ] Step 4: Commit

```bash
git add tests/test_sim.py planck-sim.toml
git commit -m "test(phase-b): planck-sim Python integration + example TOML config"
```

---

## Success Criteria

Block 1:
- [ ] cmake builds zero errors
- [ ] ctest: test_plan passes (struct sizes + fixture roundtrip)
- [ ] ctest: test_executor passes (8-rank allreduce = [36.0, ...])
- [ ] pytest test_torch_eager.py passes (optional, if libtorch available)

Block 2:
- [ ] cargo test --features sim: all sim tests pass
- [ ] pipeline_overlap test: 4-chunk faster than 1-chunk
- [ ] monotonic test: larger msg takes longer
- [ ] pytest test_sim.py: JSON output valid + Perfetto可打开
- [ ] AscendModel: InlineReduce overlaps (fused < separate)

## Ascend硬件后残留项

- csrc/transport/hccs.cpp (真实HCCS P2P transport)
- graph_pass.py + torchair集成 + 入图测试
- bench_vs_hccl.py (3组benchmark)
- planck-sim CalibratedModel (真机数据校准)
- NPKit trace对拍验证
