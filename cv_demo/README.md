# 独立视觉检测轨道

这个目录实现一条可单独运行的目标检测流程。它用于验证数据接入、模型适配、批量推理、评估、可视化和运行溯源，不生成或替换论文中的任何结果。

## 设计边界

论文复现使用 `artifact/` 中的冻结证据，不执行训练或模型推理。本目录使用用户自行准备的数据和 checkpoint，所有输出写入单独的运行目录。两条轨道共享代码质量要求，但不共享数据、结果或科学状态。

## 数据流

```text
COCO 图像与标注
        ↓
按需读取和批处理
        ↓
模型构建函数 → checkpoint → DETR / DINO 适配层
        ↓
统一预测对象
        ├── 逐图像预测
        ├── COCO 结果文件
        ├── 类别感知评估
        ├── 检测框可视化
        └── 运行记录
```

## 安装

```powershell
python -m pip install -e ".[demo]"
```

仓库不会自动下载模型或数据。请在本地准备图像目录、COCO 标注文件和与模型实现匹配的 checkpoint。

## 配置

修改 `cv_demo/configs/demo.json`：

```json
{
  "backend": "detr",
  "checkpoint": "D:/models/demo.pt",
  "images": "D:/datasets/coco/val2017",
  "annotations": "D:/datasets/coco/annotations/instances_val2017.json",
  "output": "runs/cv_demo",
  "device": "auto",
  "batch_size": 2,
  "score_threshold": 0.5,
  "iou_threshold": 0.5,
  "seed": 7,
  "category_id_map": {},
  "labels": {}
}
```

`device` 可设为 `cpu`、`cuda` 或 `auto`。`category_id_map` 用于把模型输出类别映射到数据集类别，例如 `{"0": 1}`；`labels` 用于把最终类别编号映射为可视化名称，例如 `{"1": "person"}`。

## 模型接入

不同代码库的 checkpoint 结构、预处理方式和类别头并不一致，因此模型加载由外部构建函数负责。命令行以 `模块路径:函数名` 的形式指定该函数。

```python
from pathlib import Path


def build_demo_model(checkpoint: Path):
    model = ...
    processor = ...
    model.load_state_dict(...)
    return model, processor
```

构建函数必须返回 `(model, processor)`。其中 `processor` 接收图像列表并返回模型输入；`model` 的输出要求如下：

| 后端 | 分类输出 | 边界框输出 |
|---|---|---|
| DETR | `logits` | `pred_boxes` |
| DINO | `pred_logits` | `pred_boxes` |

边界框采用归一化的中心点格式，适配层会转换为像素坐标 `x1, y1, x2, y2`。

DETR 适配层使用带“无目标”类别的 softmax 解码；DINO 适配层使用 sigmoid 解码。类别编号在评估前通过配置显式对齐，避免把模型内部编号直接当作 COCO 类别编号。

### 本地 Hugging Face DETR

仓库提供一个可直接使用的本地构建函数：

```powershell
python -m pip install -e ".[huggingface]"
python -m cv_demo.inference.predict `
  --config cv_demo/configs/demo.json `
  --builder cv_demo.models.builders.huggingface_detr:build_local_model
```

`checkpoint` 应指向包含模型配置、处理器配置和权重的本地目录。加载器启用 `local_files_only=True`，不会自动访问网络。

## 运行

```powershell
python -m cv_demo.inference.predict `
  --config cv_demo/configs/demo.json `
  --builder my_backend.builders:build_demo_model
```

也可以使用 PowerShell 入口：

```powershell
./cv_demo/scripts/run_inference.ps1 `
  -Builder my_backend.builders:build_demo_model `
  -Config cv_demo/configs/demo.json
```

## 输出与解释

- `predictions.json` 保存统一预测对象，便于调试和二次分析；
- `coco_predictions.json` 使用 COCO detection result 格式，可接入官方评估工具；
- `metrics.json` 使用类别感知的贪心匹配，报告指定 IoU 阈值下的 TP、FP、FN、精确率和召回率；
- `visualizations/` 保存带类别和置信度的检测图；
- `run_record.json` 记录配置、输入文件摘要、设备、随机种子和输出清单。

固定阈值指标用于流程检查和失败样本分析，不等同于 COCO 官方 AP。需要论文级检测基准时，应使用 `coco_predictions.json` 和目标数据集的官方评估程序。

## 测试

```powershell
python -m pytest -q cv_demo/tests
```

测试不需要模型权重，覆盖数据读取、坐标转换、类别映射、类别感知匹配、结果导出、可视化尺寸保持和运行记录隐私。

## 与论文结果的关系

本目录的任何预测、指标和图像都属于独立运行结果。它们不能写入论文的冻结证据目录，不能更新冻结聚合结果，也不能被表述为论文模型的复现实验，除非未来另行公开并核验完整的模型权重、输入数据和执行协议。
