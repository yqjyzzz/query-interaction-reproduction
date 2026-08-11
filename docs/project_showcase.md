# 给新读者的一页说明

如果只给读者几分钟，按下面的顺序介绍这个项目就够了。

## 研究问题

项目从一个具体矛盾开始：local query readout 的改善，未必带来 fixed-assignment prediction-set utility 的改善。这个问题决定了论文不能只报一个 local metric，而要同时保留 local、fixed、rematched、selection 和 native 等 readout。

## 实验为什么分成四段

- **H4-D**：先看主要 deletion response；
- **Gate C**：检查记录中的 operator-transport 条件是否成立；
- **T1**：查看 hard-minus-mass 差异是在 matching 之前还是之后出现；
- **T2**：在确认 population 上做 M/P/D pairwise classification，并检查 hard positive control。

这四段不是四个互不相关的 benchmark。后一段的结果会限制前一段能怎样解释，尤其是 T2 positive control 失败时，不能把 pairwise equivalence 写成机制确认。

## 最短演示路径

```powershell
python scripts/verify_manifest.py
python scripts/run_reproduction.py
python -m pytest -q
```

第一条命令检查冻结输入；第二条重算四组 aggregate；第三条检查仓库契约。运行过程不需要 GPU、模型权重或图像数据。

## 值得打开的文件

- `scripts/run_reproduction.py`：用户入口，负责组织路径；
- `artifact/code/p1_reproduce_from_analysis_ready.py`：聚合核心；
- `artifact/code/p1_artifact_manifest.py`：SHA-256 检查；
- `tests/test_artifact_contract.py`：检查 manifest 和 evidence firewall；
- `tests/test_exact_reproduction.py`：从输入到结果 receipt 的完整测试；
- `artifact/reproduced/REPRODUCTION_VALIDATION.json`：已经归档的验证结果。

## 说明边界时要说清楚

这是 aggregate reproduction，不是重新完成原始训练实验。仓库不包含 V/F data、image pixels 和 model weights，也没有新增 inference。论文的结论仍然限定在指定 population、operator、readout 和 estimand 下。

## 一段简短介绍

> 这是一个 CPU-only 的 TMLR 论文复现仓库。它从冻结的逐图像分析表重新计算 H4-D、Gate C、T1、T2 四组 aggregate，用 SHA-256 manifest 和 pytest 检查输入与结果的一致性，不依赖模型推理。

