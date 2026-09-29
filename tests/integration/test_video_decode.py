"""Integration test for the FFmpeg-backed in-memory video decoder."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import imageio.v2 as imageio
import numpy as np

from services.api.video import decode_sampled_frames


def test_video_bytes_are_sampled_at_the_requested_rate() -> None:
    temporary = NamedTemporaryFile(suffix=".mp4", delete=False)
    path = Path(temporary.name)
    temporary.close()

    writer = imageio.get_writer(path, fps=4, codec="libx264", macro_block_size=1)
    try:
        for index in range(12):
            writer.append_data(np.full((32, 48, 3), index * 10, dtype=np.uint8))
    finally:
        writer.close()

    try:
        frames = list(decode_sampled_frames(path.read_bytes(), sample_fps=1.0))
    finally:
        path.unlink(missing_ok=True)

    assert len(frames) == 3
    assert all(frame.shape == (32, 48, 3) for frame in frames)
