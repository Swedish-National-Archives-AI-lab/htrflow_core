from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, TypeAlias

import numpy as np

from htrflow_core.utils import draw, geometry, imgproc
from htrflow_core.utils.geometry import Bbox, Mask, Polygon


LabelType: TypeAlias = Optional[Literal["text", "class", "conf"]]


@dataclass
class Segment:
    """Segment class

    Attributes:
        bbox: The bounding box of the segment relative to the input image. Defaults to None, in which case a bounding box will be computed from the mask. Required if mask is None.
        mask: The mask of the segment, if available. The mask can either be of the same shape as the input image or of the same shape as the bounding box. It will be cropped to match the size of the bounding box if needed. Defaults to None. Required if bbox is None.
        score: Segment confidence score. Defaults to None.
        class_label: Segment class label. Defaults to None.
        polygon: An approximation of the segment mask, relative to the parent.
        orig_shape: The shape of the input image.
    """  # noqa: E501

    bbox: Optional[Bbox] = None
    mask: Optional[Mask] = None
    score: Optional[float] = None
    class_label: Optional[str] = None
    polygon: Polygon = field(init=False)
    orig_shape: Optional[tuple[int, int]] = None

    def __post_init__(self):
        """Post-initialization to compute derived attributes like polygon from mask or bbox."""

        if self.bbox is None and self.mask is None:
            raise ValueError("Cannot instantiate Segment without bbox or mask")

        if self.mask is not None:
            self.polygon = geometry.mask2polygon(self.mask)
            if self.bbox is None:
                self.bbox = geometry.mask2bbox(self.mask)

            # Crop mask to bounding box if needed
            x1, x2, y1, y2 = self.bbox
            mask_h, mask_w = self.mask.shape[:2]
            if mask_h != y2 - y1 or mask_w != x2 - x1:
                self.mask = imgproc.crop(self.mask, self.bbox)
        else:
            self.polygon = geometry.bbox2polygon(self.bbox)

        self.bbox = Bbox(*self.bbox)

    @classmethod
    def from_bbox(cls, bbox: Bbox, **kwargs) -> "Segment":
        """Create a segment from a bounding box"""
        return cls(bbox=bbox, **kwargs)

    @classmethod
    def from_mask(cls, mask: Mask, **kwargs) -> "Segment":
        """Create a segment from a mask

        Args:
            mask: A binary mask, of same shape as original image.
        """
        return cls(mask=mask, **kwargs)

    @classmethod
    def from_baseline(cls, baseline, **kwargs):
        """Create a segment from a baseline"""
        raise NotImplementedError()

    @property
    def global_mask(self):
        """The segment mask relative to the original input image"""
        if self.mask is None:
            return None
        x1, x2, y1, y2 = self.bbox
        mask = np.zeros(self.orig_shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = self.mask
        return mask

    @property
    def local_mask(self):
        """The segment mask relative to the bounding box (alias for self.mask)"""
        return self.mask


@dataclass
class RecognizedText:
    """Recognized text class

    This class represents a result from a text recognition model.

    Attributes:
        texts: A sequence of candidate texts
        scores: The scores of the candidate texts
    """

    texts: Sequence[str]
    scores: Sequence[float]

    def top_candidate(self) -> str:
        """The candidate with the highest confidence score"""
        return self.texts[self.scores.index(self.top_score())]

    def top_score(self):
        """The highest confidence score"""
        return max(self.scores)


@dataclass
class Result:
    """Result class

    This class bundles segmentation and text recognition results

    Returns:
        image: The original imaage
        metadata: Metadata associated with the result
        segments: Segments (may be empty)
        texts: Texts (may be empty)
    """

    image: np.ndarray
    metadata: dict
    segments: Sequence[Segment] = field(default_factory=list)
    texts: Sequence[RecognizedText] = field(default_factory=list)

    def __post_init__(self):
        for segment in self.segments:
            segment.orig_shape = self.image.shape[:2]

    @property
    def bboxes(self) -> Sequence[Bbox]:
        """Bounding boxes relative to input image"""
        return [segment.bbox for segment in self.segments]

    @property
    def polygons(self) -> Sequence[Polygon]:
        """Polygons relative to input image"""
        return [segment.polygon for segment in self.segments]

    @property
    def class_labels(self) -> Sequence[str]:
        """Class labels of segments"""
        return [segment.class_label for segment in self.segments]

    @classmethod
    def text_recognition_result(cls, image: np.ndarray, metadata: dict, text: RecognizedText) -> "Result":
        """Create a text recognition result

        Arguments:
            image: The original image
            metadata: Result metadata
            text: The recognized text

        Returns:
            A Result instance with the specified data and no segments.
        """
        return cls(image, metadata, texts=[text])

    @classmethod
    def segmentation_result(cls, image: np.ndarray, metadata: dict, segments: Sequence[Segment]) -> "Result":
        """Create a segmentation result

        Arguments:
            image: The original image
            metadata: Result metadata
            segments: The segments

        Returns:
            A Result instance with the specified data and no texts.
        """
        return cls(image, metadata, segments=segments)

    def plot(self, filename: Optional[str] = None, labels: LabelType = None) -> np.ndarray:
        """Plot results

        Plots the segments on the input image. If the result doesn't
        have any segments, this method will just return the original
        input image.

        Arguments:
            filename: If given, save the plotted results to `filename`
            labels: If given, plot a label of each segment. Available
                options for labels are:
                    "class": the segment class assigned by the
                        segmentation model
                    "text": the text associated with the segment
                    "conf": the segment's confidence score rounded
                        to four digits

        Returns:
            An annotated version of the original input image.
        """
        match labels:
            case "text":
                labels = [text.top_candidate() for text in self.texts]
            case "class":
                labels = self.class_labels
            case "conf":
                labels = [f"{segment.score:.4}" for segment in self.segments]
            case _:
                labels = []

        img = draw.draw_bboxes(self.image, self.bboxes, labels=labels)

        if filename:
            imgproc.write(filename, img)

        return img

    def reorder(self, index: Sequence[int]) -> None:
        """Reorder result

        Example: Given a `Result` with three segments s0, s1 and s2,
        index = [2, 0, 1] will put the segments in order [s2, s0, s1].

        Arguments:
            index: A list of indices representing the new ordering.
        """
        if self.segments:
            self.segments = [self.segments[i] for i in index]
        if self.texts:
            self.texts = [self.texts[i] for i in index]

    def drop(self, index: Sequence[int]) -> None:
        """Drop segments from result

        Example: Given a `Result` with three segments s0, s1 and s2,
        index = [0, 2] will drop segments s0 and s2.

        Arguments:
            index: Indices of segments to drop
        """
        keep = [i for i in range(len(self.segments)) if i not in index]
        self.segments = [self.segments[i] for i in keep]
        self.texts = [self.texts[i] for i in keep]
