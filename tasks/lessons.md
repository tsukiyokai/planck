# Planck Lessons Learned

## 2026-03-19: Brainstorming Session

### Lesson 1: 独立执行器 vs ACL Graph协同

最初设计Planck为独立于ACL Graph的执行系统(Rust编译plan -> C++ executor执行)。深入调研后发现HCCL通信算子已经可以在ACL Graph内部执行。独立执行器会打断graph,反而降低性能。

修正: Planck的主路径改为ACL Graph的增强层(torchair graph pass + custom ops),standalone executor降级为ACL Graph不兼容场景的备选。

教训: 在设计新系统前,必须深入理解目标平台的现有执行模型。不要假设,要验证。

### Lesson 2: KV cache transfer的优先级

最初把KV cache transfer放在v0.1的"不做"列表,同时又说它是"最大的差异化机会"。用户指出了这个矛盾。

修正: 将KV cache pipeline transfer纳入v0.1 scope(通过standalone executor,因为它不需要ACL Graph)。

教训: 新项目应该先攻无人区(HCCL没优化的KV cache transfer),而非在巨头最强的战场(AllReduce kernel调优)硬碰硬。

### Lesson 3: PyO3 vs C++26反射

用户想用C++26反射替代pybind。分析后发现:
1. C++26 P2996编译器支持还在实验阶段
2. 如果Plan Compiler在Rust,Python绑定用PyO3是Rust->Python直达,C++26反射反而多一跳
3. PyO3已经在Polars/pydantic-core等项目中生产验证

教训: "最新特性"有两种——"标准刚发布编译器没跟上"的前沿 vs "设计理念领先工程已成熟"的前沿。实战项目选后者。

### Lesson 4: Rust性能的真正杠杆

高性能Rust编程在Planck中的价值不是"让通信更快"(通信在C++/硬件层),而是解锁三个质变能力:
1. 更深的搜索 -> 更优的plan -> 间接加速每步
2. 微秒级实例化 -> per-request JIT plan
3. 毫秒级重编译 -> 在线自适应

教训: 优化要优化在杠杆点上,不是在不在瓶颈上的代码上做微优化。
