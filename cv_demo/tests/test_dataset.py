from pathlib import Path

from cv_demo.datasets import CocoDetectionDataset


def test_coco_metadata_reader() -> None:
    annotations = Path(__file__).parent / "fixtures" / "instances.json"
    dataset = CocoDetectionDataset(annotations, annotations.parent)
    sample = dataset.sample(0)
    assert sample.image_id == 3
    assert sample.file_name == "0003.jpg"
    assert sample.boxes == ((1.0, 2.0, 6.0, 8.0),)
