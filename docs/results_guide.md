# 结果怎么看

结果以 JSON 保存，方便直接检查，不额外生成一套可能和论文表格不一致的格式化数字。

## H4-D

打开 `artifact/reproduced/H4_D_REPRODUCED.json`。文件分别记录两个模型在局部、目标、固定分配、重新匹配、原生、焦点、溢出、匹配和选择读出上的估计值与自助法区间。

## Gate C

`GATE_C_REPRODUCED.json` 按剂量给出结果和联合判定。未通过的是这组预先记录的组合条件，不是“所有算子都失败”。

## T1

`T1_REPRODUCED.json` 给出配对图像对比和同时区间。它回答的是差异出现在哪个读出阶段，不是检测器的独立因果分解。

## T2

`T2_REPRODUCED.json` 同时记录 M、P、D 的配对状态和 hard 对照 H。看到 `T2_INCONCLUSIVE_PRECISION` 时，还要一起查看 `positive_control_pass` 和各端点状态。单看终止标签会丢掉这项设计信息。

## 数值校验

`REPRODUCTION_VALIDATION.json` 把新生成的四组聚合结果与 `artifact/expected/` 比较，容差为 `1e-12`。它能发现数字漂移，但不会替代对论文主张的人工审阅。

H4-D 的原生摘要和 T0 的统一原生端点不属于同一估计目标，不能跨表比较。
