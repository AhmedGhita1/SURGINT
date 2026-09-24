"""
Transform and preprocessing tests
=================================

tests the conversion between frame space and model input space.

Coverage:
- letterbox:     aspect ratio, pad placement, pad color
- boxes:         forward scaling, clipping, cxcywh conversion, inversion
- pixel_values:  shape, dtype, rescaling
- Transform:     optional annotations, train and inference modes, round trip
"""

import numpy as np
import torch

from surgint.model.transform import (
    PAD_COLOR,
    Transform,
    letterbox,
    letterbox_boxes,
    to_canvas_xyxy,
    to_normalized_cxcywh,
    to_pixel_values,
    unletterbox_boxes,
)

INPUT_SIZE = [1024, 576]
WIDTH, HEIGHT = INPUT_SIZE

RTOL = 1e-5   # float32 geometry


def blank(height: int = 720, width: int = 1280) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_letterbox():
    """aspect-preserving resize onto a fixed canvas"""

    # canvas takes the requested size whatever the frame was
    canvas, _ = letterbox(blank(), WIDTH, HEIGHT)
    assert canvas.shape == (HEIGHT, WIDTH, 3), "canvas must match the input size"
    assert canvas.dtype == np.uint8, "canvas stays uint8 until it becomes pixel values"

    # scale is the smaller of the two ratios
    # 1280x720: 1024/1280 and 576/720 both give 0.8
    _, uniform = letterbox(blank(720, 1280), WIDTH, HEIGHT)
    assert uniform == 0.8, f"expected 0.8, got {uniform}"

    # 1600x720: 1024/1600 = 0.64 is smaller than 576/720 = 0.8
    _, limited = letterbox(blank(720, 1600), WIDTH, HEIGHT)
    assert limited == 0.64, f"expected 0.64, got {limited}"

    # both axes take the same scale, so the ratio holds to within pixel rounding
    canvas, scale = letterbox(blank(720, 1600), WIDTH, HEIGHT)
    content_height, content_width = round(720 * scale), round(1600 * scale)
    assert np.allclose(content_width / content_height, 1600 / 720, rtol=1e-2), "aspect ratio drifted"

    # content sits top left, the remainder is padded
    frame = blank(720, 1600)
    frame[..., 0] = 255
    canvas, scale = letterbox(frame, WIDTH, HEIGHT)
    content_height = round(720 * scale)
    assert tuple(canvas[0, 0]) == (255, 0, 0), "content must start at the top left"
    assert tuple(canvas[content_height - 1, 0]) == (255, 0, 0), "content must fill to its last row"
    assert tuple(canvas[content_height, 0]) == (PAD_COLOR,) * 3, "padding starts below the content"
    assert tuple(canvas[-1, -1]) == (PAD_COLOR,) * 3, "the far corner is padding"

    # a frame that already matches the canvas ratio gets no padding
    frame = blank(720, 1280)
    frame[..., 0] = 255
    canvas, _ = letterbox(frame, WIDTH, HEIGHT)
    assert tuple(canvas[-1, -1]) == (255, 0, 0), "16:9 into 16:9 should not pad"

    # pad color is configurable, since it travels with the checkpoint
    canvas, _ = letterbox(blank(720, 1600), WIDTH, HEIGHT, 0)
    assert tuple(canvas[-1, -1]) == (0, 0, 0), "pad color was ignored"


def test_box_geometry():
    """forward and inverse box conversions"""

    boxes = np.array([[100.0, 200.0, 300.0, 400.0]])

    # every coordinate takes the same scale
    assert letterbox_boxes(boxes, 0.8).tolist() == [[80.0, 160.0, 240.0, 320.0]], "scaling is wrong"

    # scaling and inverting returns the original box
    _, scale = letterbox(blank(777, 333), WIDTH, HEIGHT)
    recovered = unletterbox_boxes(letterbox_boxes(boxes, scale), scale)
    assert np.allclose(recovered, boxes, rtol=RTOL, atol=1e-4), f"round trip drifted: {recovered}"

    # an image with no annotations keeps the (0, 4) shape
    empty = np.empty((0, 4))
    assert letterbox_boxes(empty, 0.8).shape == (0, 4), "empty boxes lost their shape"
    assert unletterbox_boxes(empty, 0.8).shape == (0, 4), "empty boxes lost their shape"

    # each axis is normalized by the canvas side it lies on, not by one number
    # xyxy [80, 160, 240, 320] is centred at (160, 240) and is 160x160
    cxcywh = to_normalized_cxcywh(np.array([[80.0, 160.0, 240.0, 320.0]]), WIDTH, HEIGHT)
    expected = [160 / WIDTH, 240 / HEIGHT, 160 / WIDTH, 160 / HEIGHT]
    assert np.allclose(cxcywh[0], expected, rtol=RTOL), f"expected {expected}, got {cxcywh[0]}"

    # geometry stays float32 so the model boundary needs no conversion
    assert letterbox_boxes(boxes, 0.8).dtype == np.float32, "letterbox_boxes upcast"
    assert unletterbox_boxes(boxes, 0.8).dtype == np.float32, "unletterbox_boxes upcast"
    assert to_normalized_cxcywh(boxes, WIDTH, HEIGHT).dtype == np.float32, "cxcywh upcast"



def test_pixel_values():
    """uint8 HWC canvases to the float CHW batch"""

    canvases = [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(2)]
    pixel_values = to_pixel_values(canvases)
    assert pixel_values.shape == (2, 3, HEIGHT, WIDTH), "batch and channels are in the wrong order"
    assert pixel_values.dtype == torch.float32, "model input must be float32"

    # rescaling maps 0..255 onto 0..1
    white = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    assert to_pixel_values([white]).max().item() == 1.0, "255 must become 1.0"
    assert to_pixel_values([np.zeros_like(white)]).max().item() == 0.0, "0 must stay 0.0"

    # channels move to the front without being reordered
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[..., 1] = 255
    pixel_values = to_pixel_values([canvas])
    assert pixel_values[0, 0].max().item() == 0.0, "red channel should be empty"
    assert pixel_values[0, 1].max().item() == 1.0, "green channel should be full"
    assert pixel_values[0, 2].max().item() == 0.0, "blue channel should be empty"

    print("pixel values passed")


def test_transform():

    transform = Transform(INPUT_SIZE)

    # inference passes a frame alone and gets the image half
    result = transform(blank())
    assert set(result) == {"pixel_values", "scale", "frame_size"}, f"unexpected keys {sorted(result)}"
    assert result["pixel_values"].shape == (3, HEIGHT, WIDTH), "a sample carries no batch dimension"
    assert result["scale"] == 0.8, "scale must survive for the inverse"
    assert result["frame_size"] == (720, 1280), "frame size must survive for the clip"

    # training passes the annotations too
    boxes = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
    result = transform(blank(), boxes, np.array([2], dtype=np.int64))
    assert set(result) == {"pixel_values", "scale", "frame_size", "boxes", "class_ids"}
    assert result["class_ids"].tolist() == [2], "class ids pass through untouched"

    # frame xyxy [100, 200, 300, 400] at scale 0.8 is canvas [80, 160, 240, 320]
    expected = [160 / WIDTH, 240 / HEIGHT, 160 / WIDTH, 160 / HEIGHT]
    assert np.allclose(result["boxes"][0], expected, rtol=RTOL), \
        f"expected {expected}, got {result['boxes'][0]}"

    # a box running past the frame edge is clipped, not dropped
    result = transform(blank(), np.array([[1200.0, 600.0, 1600.0, 1000.0]], dtype=np.float32), np.array([0]))
    assert result["boxes"].max() <= 1.0, "clipped boxes must stay inside the canvas"

    # an image with no annotations is not an error
    result = transform(blank(), np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64))
    assert result["boxes"].shape == (0, 4), "empty annotations lost their shape"

    # a box entirely outside the frame collapses under clipping
    try:
        transform(blank(), np.array([[1400.0, 800.0, 1500.0, 900.0]], dtype=np.float32), np.array([0]))
        assert False, "should raise ValueError for a box that clips to zero area"
    except ValueError:
        pass

    # a box that was already degenerate in the annotation
    try:
        transform(blank(), np.array([[10.0, 10.0, 10.0, 60.0]], dtype=np.float32), np.array([0]))
        assert False, "should raise ValueError for a zero-width annotation"
    except ValueError:
        pass

    # pad color and rescale factor come from the checkpoint, so they must be honoured
    configured = Transform(INPUT_SIZE, pad_color=0, rescale_factor=1.0)
    pixel_values = configured(blank(720, 1600))["pixel_values"]
    assert pixel_values[:, -1, -1].max().item() == 0.0, "pad color was ignored"


def test_postprocess():
    """canvas boxes back to frame pixels"""

    transform = Transform(INPUT_SIZE)
    result = transform(blank())

    # to_canvas_xyxy is the inverse of to_normalized_cxcywh
    canvas_boxes = to_canvas_xyxy(np.array([[160 / WIDTH, 240 / HEIGHT, 160 / WIDTH, 160 / HEIGHT]]),
                                  WIDTH, HEIGHT)
    assert np.allclose(canvas_boxes, [[80.0, 160.0, 240.0, 320.0]], rtol=RTOL), f"got {canvas_boxes}"

    # normalized cxcywh straight from the model back to frame pixels;
    # canvas [80, 160, 240, 320] at scale 0.8 is frame [100, 200, 300, 400]
    frame_boxes = transform.postprocess(
        np.array([[160 / WIDTH, 240 / HEIGHT, 160 / WIDTH, 160 / HEIGHT]]),
        result["scale"],
        result["frame_size"],
    )
    assert np.allclose(frame_boxes, [[100.0, 200.0, 300.0, 400.0]], rtol=RTOL), f"got {frame_boxes}"
    assert frame_boxes.dtype == np.float32, "predictions must be float32"

    # predictions reaching outside the frame are clipped to it
    frame_boxes = transform.postprocess(
        np.array([[0.5, 0.5, 4.0, 4.0]]), result["scale"], result["frame_size"]
    )
    assert frame_boxes.tolist() == [[0.0, 0.0, 1280.0, 720.0]], "clipping to the frame failed"

    # full round trip on a frame that actually pads
    boxes = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
    result = transform(blank(720, 1600), boxes, np.array([0]))
    recovered = transform.postprocess(result["boxes"], result["scale"], result["frame_size"])

    # one canvas pixel is 1/0.64 frame pixels, so the round trip lands within that
    assert np.allclose(recovered, boxes, rtol=RTOL, atol=1e-2), f"round trip drifted: {recovered}"

