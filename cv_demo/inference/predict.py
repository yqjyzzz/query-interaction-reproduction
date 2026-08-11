"""独立视觉检测轨道的批量推理入口。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from cv_demo.datasets import CocoDetectionDataset
from cv_demo.evaluation import combine_statistics, precision_recall_at_iou, to_coco_results
from cv_demo.inference.postprocess import remap_category_ids
from cv_demo.inference.run_record import build_run_record
from cv_demo.inference.visualize import save_detections
from cv_demo.models.contracts import GroundTruth
from cv_demo.models.detr_wrapper import DetrWrapper
from cv_demo.models.dino_wrapper import DinoWrapper
from cv_demo.models.loading import build_model


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"checkpoint", "images", "annotations", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing: {', '.join(missing)}")
    return config


def build_detector(config: dict[str, Any], builder_spec: str) -> Any:
    model, processor = build_model(builder_spec, config["checkpoint"])
    common = {
        "device": resolve_device(str(config.get("device", "auto"))),
        "score_threshold": float(config.get("score_threshold", 0.5)),
    }
    if config.get("backend", "detr").lower() == "dino":
        return DinoWrapper(model, processor, **common)
    return DetrWrapper(model, processor, **common)


def run(config: dict[str, Any], builder_spec: str, *, config_path: str | Path) -> Path:
    seed_everything(int(config.get("seed", 7)))
    dataset = CocoDetectionDataset(config["annotations"], config["images"])
    detector = build_detector(config, builder_spec)
    output_root = Path(config["output"])
    output_root.mkdir(parents=True, exist_ok=True)
    visualization_root = output_root / "visualizations"
    labels = {int(key): str(value) for key, value in config.get("labels", {}).items()}
    category_id_map = {
        int(key): int(value) for key, value in config.get("category_id_map", {}).items()
    }
    records: list[dict[str, Any]] = []
    coco_records: list[dict[str, object]] = []
    metric_rows: list[dict[str, float | int]] = []
    iou_threshold = float(config.get("iou_threshold", 0.5))
    for indices in dataset.iter_batches(int(config.get("batch_size", 1))):
        images = [dataset.load_image(index) for index in indices]
        predictions = detector.predict_batch(images)
        for index, image, raw_detections in zip(indices, images, predictions, strict=True):
            detections = remap_category_ids(raw_detections, category_id_map)
            sample = dataset.sample(index)
            records.append(
                {
                    "image_id": sample.image_id,
                    "file_name": sample.file_name,
                    "detections": [
                        {
                            "box": list(detection.box),
                            "score": detection.score,
                            "label": detection.label,
                        }
                        for detection in detections
                    ],
                }
            )
            coco_records.extend(to_coco_results(sample.image_id, detections))
            targets = [
                GroundTruth(box=box, label=label)
                for box, label in zip(sample.boxes, sample.labels, strict=True)
            ]
            image_metrics = precision_recall_at_iou(
                detections,
                targets,
                iou_threshold=iou_threshold,
            )
            metric_rows.append({"image_id": sample.image_id, **image_metrics})
            save_detections(
                image,
                detections,
                visualization_root / sample.file_name,
                labels=labels,
            )
    predictions_path = output_root / "predictions.json"
    predictions_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    coco_path = output_root / "coco_predictions.json"
    coco_path.write_text(json.dumps(coco_records, indent=2) + "\n", encoding="utf-8")
    metrics_path = output_root / "metrics.json"
    metrics_payload = {
        "method": "class_aware_greedy_matching",
        "note": "用于流程检查，不替代 COCO 官方 AP 评估。",
        "iou_threshold": iou_threshold,
        "aggregate": combine_statistics(metric_rows),
        "per_image": metric_rows,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_record = build_run_record(
        config_path=config_path,
        config=config,
        builder_spec=builder_spec,
        resolved_device=detector.device,
        image_count=len(dataset),
    )
    run_record["outputs"] = [
        predictions_path.name,
        coco_path.name,
        metrics_path.name,
        "visualizations/",
    ]
    (output_root / "run_record.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return predictions_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--builder", required=True, help="module.path:callable")
    args = parser.parse_args()
    path = run(load_config(args.config), args.builder, config_path=args.config)
    print(json.dumps({"status": "PASS_CV_DEMO_INFERENCE", "predictions": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
