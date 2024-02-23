import random
from typing import List

import cv2
import lorem  # type: ignore
import numpy as np

from htrflow_core.results import RecognitionResult, Segment, SegmentationResult


"""
This module contains dummy models
"""


class SegmentType:
    MASK = "mask"
    BBOX = "bbox"


# TODO add so metadata works like this for the dummies:
# https://github.com/Swedish-National-Archives-AI-lab/htrflow_core/blob/47fde48b12b140ac52c50f5488d71318c109bf28/src/htrflow_core/models/ultralytics/yolo.py#L11
class SegmentationModel:
    def __init__(self, segment_type: str = "mask") -> None:
        self.segment_type = segment_type

    def __call__(self, images: list[np.ndarray]) -> list[SegmentationResult]:
        metadata = generate_metadata(self)

        results = []
        for image in images:
            n_segments = random.randint(1, 5)
            h, w, _ = image.shape
            segments = []
            for _ in range(n_segments):
                score = random.random()
                if self.segment_type == SegmentType.MASK:
                    mask = randommask(h, w)
                    segments.append(Segment.from_mask(mask, score, randomlabel()))
                else:
                    bbox = randombox(h, w)
                    segments.append(Segment.from_bbox(bbox, score, randomlabel()))

            results.append(SegmentationResult(metadata, image, segments))
        return results


class RecognitionModel:
    def __call__(self, images: list[np.ndarray]) -> list[RecognitionResult]:
        metadata = generate_metadata(self)
        n = 2
        return [
            RecognitionResult(
                metadata=metadata,
                texts=[lorem.sentence() for _ in range(n)],
                scores=[random.random() for _ in range(2)],
            )
            for _ in images
        ]


def generate_metadata(model: SegmentationModel | RecognitionModel) -> dict:
    """Generates metadata for a given model."""
    model_name = model.__class__.__name__
    return {"model_name": model_name}


def randombox(h: int, w: int) -> List[int]:
    """Makes a box that is a height/3-by-width/4 and places it at a random location"""
    x = random.randrange(0, 4 * w // 5)
    y = random.randrange(0, 5 * h // 6)
    return [x, x + w // 5, y, y + h // 6]


def randomlabel() -> str:
    return "region"


def randommask(h: int, w: int) -> np.ndarray:
    """Makes an elliptical mask and places it at a random location"""
    mask = np.zeros((h, w), np.uint8)
    x, y = random.randrange(0, w), random.randrange(0, h)
    cv2.ellipse(mask, (x, y), (w // 8, h // 8), 0, 0, 360, color=(255,), thickness=-1)
    return mask
