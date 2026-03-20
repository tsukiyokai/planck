# Planck v0.1 Phase B macOS Design

> Phase B在macOS上最大化进展，无需Ascend硬件。

## Scope

两个并行Block:

```
               +-- Block 1: C++ Execution + PyTorch Eager --+
context ----> -|                                             |---> report
               +-- Block 2: planck-sim (Rust DES)        ---+
```

Block 1: 验证功能正确性 (数据对不对)
Block 2: 验证调度质量 (timeline合不合理)

不做: graph_pass.py / torchair集成 / 入图测试 / bench_vs_hccl.py

---

## Block 1: C++ Execution Layer + PyTorch Eager

### 目标

验证完整eager路径: Python torch.ops.planck.* -> C++ Executor -> MockTransport -> 正确结果

### 节点结构

```
gen_fixtures -> impl_cpp_ffi -> gate_ffi -> impl_executor -> gate_exec -> impl_torch_eager -> gate_eager
```

### gen_fixtures (shell节点)

Python脚本用PlanCompiler生成8个rank的plan bytes:

```python
import planck
compiler = planck.PlanCompiler.hccs_8card()
for rank in range(8):
    plan = compiler.compile_allreduce(256, my_rank=rank, pipeline_chunks=1)  # 256 bytes, 匹配Rust仿真测试
    open(f"csrc/test/fixtures/plan_rank{rank}.bin", "wb").write(plan.to_bytes())
```

前置: maturin develop
产出: csrc/test/fixtures/*.bin

### impl_cpp_ffi (~250行)

文件:
- csrc/include/planck/plan.h -- pragma pack(1)镜像Rust repr(C), static_assert
- csrc/include/planck/transport.h -- Transport抽象基类 (put/signal/wait/sync). put()签名: put(dst_rank, local_src, size, remote_buf_idx, offset) — 用buffer index而非裸指针, 由Transport实现负责寻址
- csrc/include/planck/executor.h -- Executor类声明
- csrc/transport/mock.cpp -- MockWorld: 共享内存 + mutex/condvar同步
- csrc/CMakeLists.txt -- 基础构建, find_package(Torch QUIET)可选
- csrc/test/test_util.h -- ~30行测试宏 (CHECK/CHECK_EQ/RUN)
- csrc/test/test_plan.cpp -- struct size断言 + fixture roundtrip

Gate: cmake --build . && ctest -R test_plan

### impl_executor (~300行)

文件:
- csrc/executor/engine.cpp -- for-loop遍历OpEntry, 处理9种opcode
- csrc/test/test_executor.cpp -- 8线程并行仿真

8-rank仿真方案: 多线程 + MockWorld共享内存

```cpp
class MockWorld {
    struct RankBuffers { input, output, scratch vectors };
    RankBuffers  ranks[8];
    mutex        mtx;
    condition_variable cv;
    int          signals[8][8];  // signals[src][dst]
};
```

每个rank在独立thread中跑完整Executor::execute():
- put(dst_rank, local_src, size, remote_buf_idx, offset): 通过MockWorld查找dst_rank的buffer地址, memcpy写入
- signal(dst_rank): 原子递增 signals[my_rank][dst_rank] + notify_all
- wait(src_rank): condvar等待 signals[src_rank][my_rank] > 0, 然后递减

远端buffer寻址: MockWorld持有所有rank的buffer指针表。Transport::put()的第4个参数是远端buffer index(不是本地指针), MockWorld用 world.ranks[dst_rank].resolve_buf(remote_buf_idx) 获取真实地址后memcpy。

验证: 输入rank r = [r+1, r+1, ...] (64个float32), 输出全部 = [36.0, ...] (sum 1..8)

关键: WaitReducePut的_pad字段存储put的远端buffer index。executor对WaitReducePut的处理:
1. Wait: transport.wait(op.dst_rank)
2. Reduce: dst_buf[j] += src_buf[j] (本地操作)
3. Put: transport.put(op.dst_rank, dst_buf, size, op._pad, 0) — 用_pad而非dst_buf寻址远端

WaitReduceCopy的_pad: reduce目标是buffer[_pad], copy源也是buffer[_pad], copy目标是dst_buf。

Gate: ctest -R test_executor

### impl_torch_eager (~200行)

文件:
- csrc/torch_binding.cpp -- TORCH_LIBRARY(planck, m) 注册custom ops
- python/planck/ops.py -- FakeTensor注册 (torch.library.register_fake)
- tests/test_torch_eager.py -- eager mode调用验证

Eager路径:
```
torch.ops.planck.pipelined_allreduce(a, b, plan_key)
  -> C++ binding -> PlanCache获取plan bytes -> Executor::execute(plan, config)
  -> MockTransport执行 -> 返回结果tensor
```

前提: pip install torch (CPU版)
Gate: pytest tests/test_torch_eager.py

### 测试框架

纯assert + test_util.h (~30行), 不引入Catch2/GTest。
理由: planck-core Rust侧用内建#[test], C++侧保持对称极简。
cmake ctest提供test discovery。
迁移到Ascend环境时零负担。

### 构建

独立cmake, 不集成corrosion。Rust和C++通过plan bytes文件交互, 构建层无耦合。
find_package(Torch QUIET): 有libtorch时编译torch_binding, 没有时跳过。

---

## Block 2: planck-sim 内嵌仿真器

### 目标

Planck编译器的开发反馈工具。给plan加上时间维度, 输出Chrome Trace timeline。
回答 "这个schedule好不好", 不回答 "真实硬件跑多快"。

### 定位

不是通用CCL仿真平台。是编译器内嵌的schedule验证工具。
将来有Ascend真机数据校准后, 可提升精度, 但这不是v0.1目标。

### 业界方法论基础

| 设计决策                    | 来源            | 理由                                |
|:---------------------------|:---------------|:------------------------------------|
| chunk级analytical建模       | Echo论文        | 8%误差, 速度快1000倍于packet级       |
| 4层事件层次                 | NPKit事件层次    | Perfetto缩放查看                    |
| TimingModel trait可插拔     | ASTRA-sim 3后端  | SimpleModel扫参, AscendModel精细    |
| TOML配置驱动               | Days (Rust DES)  | 零代码调参                          |
| Chrome Trace + NPKit兼容格式 | NPKit + SimAI   | 将来真机trace对拍                   |

### Ascend硬件特征建模 (来自HCOMM源码)

| 要素                  | 建模方式                              | 来源                    |
|:---------------------|:--------------------------------------|:------------------------|
| 3轮Notify握手         | 3 * link.latency_us                  | phase2-hcomm-platform   |
| GET模式(HCCS)额外延迟  | 2 * latency + data_time (vs 1 * PUT) | prim_rules.cc           |
| SQE队列深度           | 2048/stream, 超限排队                 | stream_pub.h            |
| 门铃批处理             | 每10 WQE一次doorbell overhead         | send_recv_executor      |
| InlineReduce重叠      | max(reduce_time, put_time) 非相加     | dispatcher_pub.h        |
| CQ轮询间隔            | 10us固定overhead                     | transport_roce.cc       |

### 架构

planck-core/src/sim/ 目录, 不新建subcrate。
通过feature flag隔离外部依赖, 保持默认零依赖:

```toml
# crates/planck-core/Cargo.toml
[features]
sim = ["toml", "serde"]

[dependencies]
toml  = { version = "0.8", optional = true }
serde = { version = "1", features = ["derive"], optional = true }
```

`cargo build`默认不编译sim模块, `cargo build --features sim`或`cargo test --features sim`启用。

文件结构:

```
sim/
  mod.rs       公开API: simulate(plans, config) -> Trace
  engine.rs    DES核心: EventQueue(BinaryHeap) + Clock + 主循环
  link.rs      链路状态: active flows, 带宽竞争, GET/PUT模式
  timing.rs    TimingModel trait + SimpleModel + AscendModel
  trace.rs     Chrome Trace JSON: 4层嵌套 (Collective/Chunk/Op/HwAction)
  config.rs    TOML配置解析 (含size parser: "256MB" -> 268435456)
```

### DES引擎

```rust
struct Event {
    time:   f64,       // 微秒
    rank:   u16,
    stream: u8,
    kind:   EventKind, // OpStart / OpEnd / PutArrive / SignalArrive / NotifyRound
}

struct Simulator {
    queue:    BinaryHeap<Reverse<Event>>,
    clock:    f64,
    links:    Vec<LinkState>,
    streams:  Vec<Vec<StreamState>>,  // [rank][stream]
    trace:    Trace,
    model:    Box<dyn TimingModel>,
}
```

主循环: pop最小时间事件 -> 根据EventKind调度 -> 生成后续事件push回队列 -> 记录trace event

### TimingModel trait

```rust
trait TimingModel {
    fn put_time(&self, link: &Link, size: usize) -> f64;
    fn notify_time(&self, link: &Link) -> f64;
    fn reduce_time(&self, size: usize) -> f64;
    fn inline_reduce_put_time(&self, link: &Link, size: usize) -> f64;
    fn kernel_launch_overhead(&self) -> f64;
}
```

SimpleModel: alpha-beta公式, 一行计算, 毫秒出结果。适合批量扫参。
AscendModel: 3轮notify + GET模式 + SQE队列 + InlineReduce重叠 + doorbell批处理。

### Chrome Trace输出

4层事件层次:

```
Collective (AllReduce 256MB)            -- pid=rank, tid=0
  Pipeline_Chunk (chunk 0/1/2/3)        -- B/E嵌套
    Op (Put/Wait/Reduce/...)            -- X event
      HwAction (notify_handshake/dma)   -- X event, cat="hw"
```

rank间数据流: flow event (ph:"s"/"f") 画rank间Put箭头
带宽利用率: counter event (ph:"C") 每条链路的瞬时利用率

### TOML配置

```toml
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

### PyO3接口

```python
import planck

# 方式1: 编程式
compiler = planck.PlanCompiler.hccs_8card()
plans = [compiler.compile_allreduce(256<<20, rank, pipeline_chunks=4) for rank in range(8)]
trace = planck.simulate(plans)
trace.save("trace.json")

# 方式2: TOML配置
trace = planck.simulate_from_config("planck-sim.toml")
trace.save("trace.json")
```

### 验证

1. 功能: 4-chunk pipeline的chunk 0 Put和chunk 1 Reduce在时间上重叠
2. 单调: 消息越大耗时越长
3. Pipeline收益: 4-chunk估计耗时 < 1-chunk估计耗时
4. InlineReduce: WaitReducePut估计耗时 < Wait + Reduce + Put分开的估计耗时

### 估计代码量

```
engine.rs     ~200行   事件队列 + 主循环 + op调度
link.rs       ~100行   链路状态 + 带宽竞争
timing.rs     ~150行   TimingModel trait + Simple + Ascend两个实现
trace.rs      ~150行   Chrome Trace 4层序列化
config.rs     ~100行   TOML配置解析
mod.rs        ~100行   simulate() API
```

~800行Rust + ~50行PyO3绑定

---

## Ascend硬件后残留项

Block 1:
- csrc/transport/hccs.cpp -- 真实HCCS P2P transport替换mock
- hccs.cpp中InlineReduce的真实实现
- torch_binding.cpp接入真实transport

Block 2 (planck-sim):
- 用真机profiling数据校准timing参数
- CalibratedModel (第三种TimingModel)
- 与NPKit trace对拍验证

独立项:
- graph_pass.py + torchair注册 + 入图测试
- bench_vs_hccl.py (3组benchmark)

---

## Changelog

- 2026-03-20: Phase B macOS设计完成
