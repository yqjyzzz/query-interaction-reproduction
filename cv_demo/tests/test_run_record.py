from pathlib import Path

from cv_demo.inference.run_record import build_run_record, describe_path


def test_run_record_hashes_inputs_without_absolute_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "demo.json"
    checkpoint_path = tmp_path / "weights.pt"
    annotations_path = tmp_path / "instances.json"
    config_path.write_text("{}", encoding="utf-8")
    checkpoint_path.write_bytes(b"weights")
    annotations_path.write_text('{"images": []}', encoding="utf-8")
    record = build_run_record(
        config_path=config_path,
        config={
            "checkpoint": str(checkpoint_path),
            "annotations": str(annotations_path),
        },
        builder_spec="example:build",
        resolved_device="cpu",
        image_count=0,
    )
    assert record["paper_artifact_written"] is False
    assert record["checkpoint"]["name"] == "weights.pt"
    assert str(tmp_path) not in str(record)


def test_directory_checkpoint_has_stable_summary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "weights.bin").write_bytes(b"weights")
    summary = describe_path(checkpoint)
    assert summary["kind"] == "directory"
    assert summary["file_count"] == 2
    assert summary["name"] == "model"
