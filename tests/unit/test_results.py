import cv2
import numpy as np
import pytest

from htrflow_core.results import Segment


@pytest.fixture
def overflowing_ellipse_mask():
    mask = np.zeros((6, 6), dtype=np.uint8)
    cv2.ellipse(mask, (3, 3), (2, 2), 0, 0, 360, 1, -1)
    return mask


@pytest.fixture
def ellipse_mask_with_fitted_bbox_tuple():
    mask = np.zeros((5, 5), dtype=np.uint8)
    cv2.ellipse(mask, (2, 2), (2, 2), 0, 0, 360, 1, -1)
    bbox = (0, 5, 0, 5)
    return mask, bbox


@pytest.mark.usefixtures("overflowing_ellipse_mask", "ellipse_mask_with_fitted_bbox_tuple")
class TestSegment:
    def test_segment_initialization_with_no_bbox_or_mask(self):
        with pytest.raises(ValueError):
            Segment()

    def test_segment_initialization_with_overflowing_mask(self, overflowing_ellipse_mask):
        segment = Segment(mask=overflowing_ellipse_mask)

        expected_bbox = (1, 5, 1, 5)
        expected_polygon = np.array([[3, 1], [1, 3], [3, 5], [5, 3]])

        assert segment.bbox.xxyy == expected_bbox, "Bbox should enclose the ellipse"
        assert np.allclose(segment.polygon, expected_polygon), "Polygon should approximate the ellipse"

    def test_segment_initialization_with_mask_and_bbox(self, ellipse_mask_with_fitted_bbox_tuple):
        given_mask, given_bbox = ellipse_mask_with_fitted_bbox_tuple
        segment = Segment(mask=given_mask, bbox=given_bbox)

        expected_polygon = np.array([[2, 0], [0, 2], [2, 4], [4, 2]])

        assert segment.bbox.xxyy == given_bbox, "Provided bbox should be used"
        assert np.array_equal(segment.mask, given_mask), "Provided mask should be used"
        assert np.array_equal(segment.polygon, expected_polygon), "Polygon should approximate the ellipse"
