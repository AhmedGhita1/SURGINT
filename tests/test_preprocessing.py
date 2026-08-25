import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from surgint.detection.preprocessing import PAD_VALUE, letterbox, letterbox_boxes, unletterbox_boxes

WIDTH, HEIGHT = 960, 544


def test_unit_letterbox_geometry():
    """🧪 Test the canvas size, the scale factor, and that nothing is stretched."""
    # 1280x720 into 960x544: width ratio 0.750, height ratio 0.756
    # the smaller one wins, so both sides shrink by 0.75 and 16:9 survives
    canvas, scale = letterbox(Image.new("RGB", (1280, 720)), WIDTH, HEIGHT)
    assert canvas.size == (WIDTH, HEIGHT)
    assert scale == 0.75
    assert round(1280 * scale) == 960
    assert round(720 * scale) == 540

    # a portrait source is limited by height instead, and still is not stretched
    _, scale = letterbox(Image.new("RGB", (720, 1280)), WIDTH, HEIGHT)
    assert scale == 544 / 1280
    assert round(1280 * scale) == 544

    # a square source into a wide canvas is limited by height too
    _, scale = letterbox(Image.new("RGB", (640, 640)), WIDTH, HEIGHT)
    assert scale == 544 / 640


def test_unit_letterbox_padding():
    """🧪 Test that the image sits top-left and the leftover space is padding."""
    red = Image.new("RGB", (1280, 720), (255, 0, 0))
    canvas, _ = letterbox(red, WIDTH, HEIGHT)

    # the image occupies rows 0..539, so its corners are still red
    assert canvas.getpixel((0, 0)) == (255, 0, 0)
    assert canvas.getpixel((959, 539)) == (255, 0, 0)

    # rows 540..543 are the 4 leftover rows, filled with PAD_VALUE
    assert canvas.getpixel((0, 540)) == (PAD_VALUE, PAD_VALUE, PAD_VALUE)
    assert canvas.getpixel((959, 543)) == (PAD_VALUE, PAD_VALUE, PAD_VALUE)

    # ⚠️ the corner matters. training and inference must pad the SAME corner,
    # or the same object lands at different coordinates in the two paths and
    # the model quietly learns an offset. top-left is the convention here.
    assert canvas.getpixel((0, 0)) != (PAD_VALUE, PAD_VALUE, PAD_VALUE)


def test_unit_box_transforms():
    """🧪 Test boxes forward to canvas pixels and back to camera pixels."""
    # forward: every coordinate is a pixel length, so all four scale
    box = np.array([[100.0, 200.0, 300.0, 400.0]])
    assert letterbox_boxes(box, 0.75).tolist() == [[75.0, 150.0, 225.0, 300.0]]

    # backward: the inverse of the same scale
    assert unletterbox_boxes(np.array([[75.0, 150.0, 225.0, 300.0]]), 0.75).tolist() == box.tolist()

    # a full round trip must land exactly where it started
    _, scale = letterbox(Image.new("RGB", (1280, 720)), WIDTH, HEIGHT)
    assert np.array_equal(unletterbox_boxes(letterbox_boxes(box, scale), scale), box)

    # and for an awkward source size where the scale is not a round number
    _, scale = letterbox(Image.new("RGB", (333, 777)), WIDTH, HEIGHT)
    roundtrip = unletterbox_boxes(letterbox_boxes(box, scale), scale)
    assert np.abs(roundtrip - box).max() < 1e-9, f"drifted by {np.abs(roundtrip - box).max()}"

    # ⚠️ forgetting the inverse at inference is the classic bug: boxes get
    # reported in canvas pixels and land on the wrong part of the frame.
    assert not np.array_equal(letterbox_boxes(box, 0.75), box)

    # a frame with no objects must not crash
    empty = np.empty((0, 4))
    assert letterbox_boxes(empty, 0.75).shape == (0, 4)
    assert unletterbox_boxes(empty, 0.75).shape == (0, 4)


if __name__ == "__main__":
    from report import TestReport

    print("🧪 Testing preprocessing...")
    report = TestReport("Preprocessing")
    for name, test in [
        ("Letterbox Geometry", test_unit_letterbox_geometry),
        ("Letterbox Padding", test_unit_letterbox_padding),
        ("Box Transforms", test_unit_box_transforms),
    ]:
        report.run(name, test)
    report.finish()
