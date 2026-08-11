# 复现契约

## 环境

- Python 3.12.x
- NumPy 1.26.4
- Pytest 8.x（仅运行测试时需要）

聚合结果复现只使用 CPU。原始模型运行曾使用 PyTorch、torchvision、SciPy 和 NVIDIA RTX 4090，但论文复现轨道不加载模型。

## 运行顺序

```text
校验 ARTIFACT_MANIFEST.json
        |
读取 analysis_ready/
        |
运行 p1_reproduce_from_analysis_ready.py
        |
与 expected/ 比较，容差 1e-12
```

标准命令由 `scripts/` 提供：

```powershell
python scripts/verify_manifest.py
python scripts/run_reproduction.py
```

结果回执会保留 `V_F_read = false` 和 `new_model_inference = false`。

## 冻结清单

`artifact/ARTIFACT_MANIFEST.json` 记录 62 个冻结文件的字节数和 SHA-256。清单自身以及可选的 `reproduced_selftest/` 不在校验范围内。如果校验失败，应先调查文件差异，不要直接重新生成清单。

## 为什么论文轨道不提供端到端推理

公开包没有图像像素和模型权重。来源文档记录了原始数据族和模型检查点摘要，但这些记录不等于资源已经重新分发。若以后开放论文模型推理，需要另行固定环境、样本选择、输入捕获和完整性审计。

独立视觉检测实现位于 `cv_demo/`。它可以接入用户提供的模型与数据，但其输出不属于论文冻结结果。
