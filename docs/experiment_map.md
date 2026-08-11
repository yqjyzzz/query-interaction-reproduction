# 实验结构

论文把同一个科学问题拆成四个有明确分工的分析。

| 组件 | 问题 | 输入 | 输出 |
|---|---|---|---|
| H4-D | hard 干预是否改变发现样本总体上的读出？ | `analysis_ready/h4_d_*.jsonl` | `expected/H4_D_SCIENTIFIC_AGGREGATE.json` |
| Gate C | 记录中的算子迁移联合条件是否在冻结剂量上通过？ | `analysis_ready/gate_c_*.jsonl` | `expected/H4_GATE_C_SCIENTIFIC_AGGREGATE.json` |
| T1 | hard-minus-mapped-mass 差异最早在哪一层出现？ | `analysis_ready/t1_*.jsonl` | `expected/T1_PRE_READOUT_LOCALIZATION.json` |
| T2 | M/P/D 是否在冻结区间内等价？强阳性对照是否通过？ | `analysis_ready/t2_*.jsonl` | `expected/T2_CONFIRMATION_INITIAL_128_AGGREGATE.json` |

## 数据流

```text
analysis-ready rows
        |
        +-- H4-D 分层自助法
        +-- Gate C 分剂量自助法
        +-- T1 配对图像同时区间
        +-- T2 配对图像、分层重采样同时区间
        |
        v
冻结聚合 JSON
        |
        v
与 artifact/expected/ 做精确比对
```

两个模型分开分析。仓库不会把 DETR 和 DINO 合并成新的汇总结果。

H4-D 是主要现象；Gate C 检查它能否沿指定算子条件迁移；T1 帮助判断差异是在读出之前还是匹配敏感读出之后出现；T2 提供确认性配对分类。T2 的强阳性对照是解释链的一部分，不是附带的质量指标。
