# Phase B Block 2: planck-sim DES Simulator

## 目标

实现Phase B impl plan的Chunk 6 + Chunk 7 Tasks 13-15:
- Task 13: timing.rs (TimingModel trait + SimpleModel + AscendModel)
- Task 14: trace.rs (Trace struct + Chrome Trace JSON)
- Task 15: sim/mod.rs集成测试 (pipeline_overlap, monotonic, trace_has_events)

## 状态: 已完成 (已验证)

Tasks 13-15在先前迭代中已全部实现。本轮重新验证: 17/17 sim测试通过。

```
cargo test -p planck-core --features sim -- sim
  17 passed; 0 failed    # 2026-03-21 re-verified
```

### 已完成产物

| Task | 文件 | 行数 | 状态 |
|:-----|:-----|-----:|:-----|
| 11 | Cargo.toml + lib.rs + sim/mod.rs + config.rs | 165 | done |
| 12 | engine.rs + link.rs | 275+58 | done |
| 13 | timing.rs | 130 | done |
| 14 | trace.rs | 96 | done |
| 15 | mod.rs集成测试(4个) | 87 | done |

### 实现 vs impl plan的差异

1. timing.rs SimpleModel: 实际有`hbm_bw_gbps`字段(可配置)，impl plan是unit struct(硬编码460)。实际更好。
2. trace.rs TraceEvent: 实际无`cat`字段，在to_json()中硬编码`"planck"`。更简洁。
3. config.rs TimingConfig: 实际省略了impl plan中的`sqe_depth`/`doorbell_batch`字段(未被任何代码引用)。
4. engine.rs WaitReducePut: 实际用`op.wait_event`获取Wait源rank，修正了impl plan中`op.dst_rank`的bug。
5. 集成测试: 实际有4个测试(多了`sim_completes_without_deadlock`)，超出plan要求的3个。

### 关键技术要点(供后续参考)

- AscendModel三个核心公式:
  - `put_time = 2*latency + size/bw` (GET模式: request+data两次延迟)
  - `notify_time = 3*latency` (HCCS 3轮握手)
  - `inline_reduce_put_time = notify + max(reduce, put)` (MTE+AIV物理隔离可重叠)
- DES引擎: BinaryHeap最小堆 + signal/wait跟踪 + LinkState公平共享带宽模型
- Chrome Trace: pid=rank, tid=stream, ph='X'完成事件, Perfetto兼容

## 下一步

Tasks 16-17 (不在当前任务范围，但是Block 2的剩余部分):
- Task 16: PyO3 `simulate()` binding — 在planck-python crate中暴露sim API
- Task 17: Python test (test_sim.py) + TOML配置示例 (planck-sim.toml)

tasks/todo.md中Phase B Chunk 6/7的checkbox未打勾，但代码已完成。
