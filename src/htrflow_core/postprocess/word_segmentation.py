from htrflow_core.results import RecognizedText, Result, Segment
from htrflow_core.utils.geometry import bbox2mask
from htrflow_core.utils.imgproc import mask
from htrflow_core.volume.volume import SegmentNode


def _simple_word_segmentation(node: SegmentNode):
    text = node.text
    words = text.split()
    pixels_per_char = node.width // len(text)
    x1, x2 = 0, 0
    bboxes = []
    for word in words:
        x2 = min(x1 + pixels_per_char * (len(word) + 1), node.width)
        bboxes.append((x1, 0, x2, node.height))
        x1 = x2
    segments = [
        Segment(mask=mask(node.mask, bbox2mask(bbox, node.mask.shape), fill=0), class_label="word") for bbox in bboxes
    ]
    texts = [RecognizedText([word], [0]) for word in words]
    return Result({}, segments=segments, texts=texts)


def simple_word_segmentation(nodes: list[SegmentNode]):
    return [_simple_word_segmentation(node) for node in nodes]
