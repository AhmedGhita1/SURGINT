"""The checkpoint the image smoke test serves.

models/ is not in the repository, so this builds one from the pretrained
weights and the label set the ontology recognizes. the geometry is smaller
than a trained checkpoint's so one frame decodes quickly on a runner's cpu.
"""

import sys
from pathlib import Path

import yaml

from surgint.model.detector import Detector
from surgint.ontology.validation import PERCEPTION_LABELS

TRAIN_CONFIG = Path("configs/train.yaml")
INPUT_SIZE = [320, 192]


def main(destination: str) -> None:
    config = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    source = config["checkpoint"]

    labels = dict(enumerate(sorted(PERCEPTION_LABELS)))
    detector = Detector.from_pretrained(source, labels, revision=config.get("revision"))
    detector.save_checkpoint(
        Path(destination),
        {
            "input_size": INPUT_SIZE,
            "pad_color": 114,
            "rescale_factor": 1 / 255,
            "source": source,
        },
    )
    print(f"wrote {len(labels)} labels to {destination}")


if __name__ == "__main__":
    main(sys.argv[1])
