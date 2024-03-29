from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, TypeAlias

import numpy as np

from htrflow_core.utils import draw, geometry, imgproc
from htrflow_core.utils.geometry import Bbox, Mask, Polygon


LabelType: TypeAlias = Literal["text", "class", "conf"] | None


class Segment:
    """Segment class

    Class representing a segment of an image, typically a result from
    a segmentation model or a detection model.

    Attributes:
        bbox: The bounding box of the segment
        mask: The segment's mask, if available. The mask is stored
            relative to the bounding box. Use the `global_mask()`
            method to retrieve the mask relative to the original image.
        score: Segment confidence score, if available.
        class_label: Segment class label, if available.
        polygon: An approximation of the segment mask, relative to the
            original image. If no mask is available, `polygon` defaults
            to a polygon representation of the segment's bounding box.
        orig_shape: The shape of the orginal input image.
    """

    bbox: Bbox
    mask: Mask | None
    score: float | None
    class_label: str | None
    polygon: Polygon
    orig_shape: tuple[int, int] | None

    def __init__(
            self,
            bbox: tuple[int, int, int, int] | Bbox | None = None,
            mask: Mask | None = None,
            score: float | None = None,
            class_label: str | None = None,
            polygon: Polygon | Sequence[tuple[int, int]] | None = None,
            orig_shape: tuple[int, int] | None = None
    ):
        """Create a `Segment` instance

        A segment can be created from a bounding box, a polygon, a mask
        or any combination of the three.

        Arguments:
            bbox: The segment's bounding box, as either a `geometry.Bbox`
                instance or as a (xmin, ymin, xmax, ymax) tuple. Required
                if `mask` and `polygon` are None. Defaults to None.
            mask: The segment's mask, either relative to the bounding box
                or relative to the original input image. If both `bbox`
                and `polygon` are None, `mask` is required and must be the
                same shape as the original input image. Defaults to None.
            score: Segment confidence score. Defaults to None.
            class_label: Segment class label. Defaults to None.
            polygon: A polygon defining the segment, relative to the input
                image. Defaults to None. Required if both `mask` and `bbox`
                are None.
            orig_shape: The shape of the orginal input image. Defaults to
                None.
        """

        # Convert polygon and bbox to Polygon and Bbox instances
        if polygon is not None:
            polygon = geometry.Polygon(polygon)
        if bbox is not None:
            bbox = geometry.Bbox(*bbox)


        match (bbox, mask, polygon):

            case (None, None, None):
                raise ValueError("Cannot create a Segment without bbox, mask or polygon")

            case (_, None, None):
                # Only bbox is given: Create a polygon from the bbox and leave the
                # mask as None.
                polygon = bbox.polygon()

            case (None, _, None):
                # Only mask is given: In this case, the mask is assumed to be aligned
                # with the original image, i.e., it has the same height and width as
                # the input image. The other attributes (bbox and polygon) can in such
                # case be inferred from the mask. After computing them, the mask is
                # converted to a local mask.
                bbox = geometry.mask2bbox(mask)
                polygon = geometry.mask2polygon(mask)
                mask = imgproc.crop(mask, bbox)

            case (None, None, _):
                # Only polygon is given: Create a bounding box from the polygon and
                # leave the mask as None.
                bbox = geometry.Polygon(polygon).bbox()

            case (_, _, None):
                # Both bbox and mask are given: Create a polygon from the mask.
                polygon = geometry.mask2polygon(mask)
                if mask.shape[:2] == (bbox.height, bbox.width):
                    polygon = polygon.move(bbox.p1)
                else:
                    mask = imgproc.crop(mask, bbox)

        self.bbox = bbox
        self.polygon = polygon
        self.mask = mask
        self.score = score
        self.class_label = class_label
        self.orig_shape = orig_shape

    def global_mask(self, orig_shape: tuple[int, int] | None = None) -> Optional[Mask]:
        """
        The segment mask relative to the original input image.

        Arguments:
            orig_shape: Pass this argument to use another original shape
                than the segment's `orig_shape` attribute. Defaults to None.
        """
        if self.mask is None:
            return None

        orig_shape = self.orig_shape if orig_shape is None else orig_shape
        if orig_shape is None:
            raise ValueError("Cannot compute the global mask without knowing the original shape.")

        x1, y1, x2, y2 = self.bbox
        mask = np.zeros(orig_shape, dtype=np.uint8)
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
    def global_masks(self) -> Sequence[Mask]:
        """Global masks relative to input image"""
        return [segment.global_mask for segment in self.segments]

    @property
    def local_mask(self) -> Sequence[Mask]:
        """Local masks relative to bounding boxes"""
        return [segment.local_mask for segment in self.segments]

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

        if self.segments:
            self.segments = [self.segments[i] for i in keep]
        if self.texts:
            self.texts = [self.texts[i] for i in keep]
