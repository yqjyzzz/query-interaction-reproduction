# 数据与模型溯源

完整记录见 [`artifact/DATA_AND_MODEL_PROVENANCE.md`](../artifact/DATA_AND_MODEL_PROVENANCE.md)。这里保留一份便于查阅的摘要。

## 样本总体

- Discovery / Gate C 使用源自 Open Images 验证集的 D 样本总体和冻结的 COCO 类别映射，每个模型检查点有 710 张完整配对图像；
- T2 使用 COCO 2017 验证集，初始分析为每个模型 128 张图，冻结总体上限为 256 张；样本选择没有读取模型结果或干预结果。

公开包保留不透明图像标识、图像摘要、分层信息和派生分析值，不包含图像像素。

## 模型检查点

- DETR-R50-500：`e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc`；
- DINO-R50、4-scale、12-epoch：`0bcd6b0c33d60ed33461ce6f02ce5797a819c7c02eb7e15b76adfb6df307955a`。

权重没有放入仓库；这些 SHA-256 仅用于标识原始来源。

## 未包含的材料

V/F 数据、图像像素、模型权重、私有凭据、完整原始质量矩阵和 T0B 原始输入回执均未公开。N1 只保留其聚合结果、审计记录和配置，作为历史可操作性边界。
