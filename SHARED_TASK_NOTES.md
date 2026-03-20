# Phase B Block 2: planck-sim DES Simulator

## Tasks 11-15 状态: 已完成

17/17 sim测试通过。详见git history。

## Chunk 7: Tasks 16-17 — 已完成

### 已完成产物

| 步骤 | 文件 | 变更 |
|:-----|:-----|:-----|
| 16.1 | `crates/planck-python/Cargo.toml` | planck-core加 `features = ["sim"]` |
| 16.2 | `crates/planck-python/src/lib.rs` | +22行: `#[pyfunction] simulate()` + module注册 |
| 16.3 | `python/planck/__init__.py` | 重导出 `simulate` |
| 17.1 | `tests/test_sim.py` | 3个测试: json有效/toml配置/pipeline更多事件 |
| 17.2 | `planck-sim.toml` | 示例配置(去掉了不存在的sqe_depth/doorbell_batch) |

### 验证结果

```
pytest tests/ -v                              → 7 passed
cargo test -p planck-core --features sim      → 17 passed
```

### 设计决策

1. simulate()接受 `Vec<PyRef<PyPlanView>>` — PyO3自动从Python list提取
2. 硬编码 `Topology::hccs_8card()` — macOS演示阶段无法探测真实拓扑
3. TOML示例去掉impl plan中的 `sqe_depth`/`doorbell_batch` — 当前SimConfig没这些字段
4. Python __init__.py 直接重导出，不加wrapper函数 — 保持简洁

### 下一步

Phase B Block 2全部完成。剩余工作需要Ascend硬件:
- csrc/transport/hccs.cpp (真实HCCS P2P transport)
- graph_pass.py + torchair集成
- bench_vs_hccl.py
- planck-sim CalibratedModel
