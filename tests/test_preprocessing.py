import numpy as np
import torch


from surgint.dataset.transform import (
    PAD_VALUE,
    letterbox,
    letterbox_boxes,
    to_pixel_values,
    unletterbox_boxes,
)


def test_letterbox_preserves_aspect_ratio():
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    canvas, scale = letterbox(frame, 1024, 576)

    assert canvas.shape == (576, 1024, 3)
    assert scale == 576 / 1280
    assert round(720 * scale) / round(1280 * scale) == 720 / 1280


def test_letterbox_pads_bottom_right():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[..., 0] = 255

    canvas, scale = letterbox(frame, 1024, 640)

    content_height = round(720 * scale)
    assert tuple(canvas[0, 0]) == (255, 0, 0)
    assert tuple(canvas[content_height - 1, -1]) == (255, 0, 0)
    assert tuple(canvas[content_height, 0]) == (PAD_VALUE, PAD_VALUE, PAD_VALUE)


def test_box_transform_roundtrip():
    boxes = np.array([[100.0, 200.0, 300.0, 400.0]])
    _, scale = letterbox(np.zeros((777, 333, 3), dtype=np.uint8), 1024, 576)

    canvas_boxes = letterbox_boxes(boxes, scale)

    assert not np.array_equal(canvas_boxes, boxes)
    assert np.abs(unletterbox_boxes(canvas_boxes, scale) - boxes).max() < 1e-9


def test_box_transform_handles_empty_input():
    empty = np.empty((0, 4))

    assert letterbox_boxes(empty, 0.8).shape == (0, 4)
    assert unletterbox_boxes(empty, 0.8).shape == (0, 4)


def test_to_pixel_values_shape_and_dtype():
    canvases = [np.zeros((576, 1024, 3), dtype=np.uint8) for _ in range(2)]

    pixel_values = to_pixel_values(canvases)

    assert pixel_values.shape == (2, 3, 576, 1024)
    assert pixel_values.dtype == torch.float32


def test_to_pixel_values_rescales_to_unit_range():
    white = np.full((576, 1024, 3), 255, dtype=np.uint8)

    assert to_pixel_values([white]).max().item() == 1.0
    assert to_pixel_values([np.zeros_like(white)]).max().item() == 0.0
