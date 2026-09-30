"""Decode a completely uploaded video into a sampled stream of RGB frames."""

from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

import imageio_ffmpeg
import numpy as np


class InvalidVideoError(ValueError):
    pass


def decode_sampled_frames(contents: bytes, sample_fps: float) -> Iterator[np.ndarray]:
    """Yield RGB frames sampled from a complete in-memory video upload."""
    decoded_frames = None
    temporary_path = None
    try:
        # The ImageIO FFmpeg adapter selects its reader from the suffix; FFmpeg
        # itself probes the bytes, so this also accepts WebM and MOV uploads.
        with NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
            temporary.write(contents)
            temporary_path = Path(temporary.name)

        decoded_frames = imageio_ffmpeg.read_frames(
            str(temporary_path),
            pix_fmt="rgb24",
            bits_per_pixel=24,
            output_params=["-vf", f"fps={sample_fps}"],
        )
        metadata = next(decoded_frames)
        width, height = metadata["size"]
        for decoded in decoded_frames:
            yield np.frombuffer(decoded, dtype=np.uint8).reshape(height, width, 3)
    except InvalidVideoError:
        raise
    except Exception as error:
        raise InvalidVideoError("video could not be decoded") from error
    finally:
        if decoded_frames is not None:
            decoded_frames.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
