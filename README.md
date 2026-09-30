![Status: Research Demo](https://img.shields.io/badge/status-research%20demo-orange) [![CI](https://github.com/AhmedGhita1/surgint-instruments/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedGhita1/surgint-instruments/actions/workflows/ci.yml) [![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10--3.12-blue)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/AhmedGhita/surgint) [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/AhmedGhita/surgint-instruments-rt-detr)

# SURGINT Instruments

*This project is part of the surgical intelligence (SURGINT) family.*

SURGINT Instruments is a research prototype that constructs and monitors surgical instrument inventories from tray videos or live camera feeds, using RT-DETR and ByteTracker for detection and tracking and ontology-backed rules for handling-procedure decision support.

> **Research demonstration only.** SURGINT Instruments has not been clinically validated, and must not be used for clinical decisions, patient care, or safety-critical instrument accounting.

---
## Architecture

![SURGINT Instruments architecture](docs/SURGINT-architecture-v2.svg)

---
## Run with Docker

Running the Docker image requires Docker, the NVIDIA Container Toolkit, and an NVIDIA driver
compatible with CUDA 12.1.

```bash
docker pull ghcr.io/ahmedghita1/surgint-instruments:1.0.0
docker run --rm --gpus all -p 7860:7860 ghcr.io/ahmedghita1/surgint-instruments:1.0.0
```

Open <http://localhost:7860> after the container reports healthy.

---
## Development

SURGINT Instruments supports Python 3.10–3.12. The `train` extra contains both training and evaluation dependencies, while `dev` includes `serve`, `train`, and the development tools. Install the complete development environment with:

```bash
python -m pip install -e ".[dev]"
```

Run the API locally with a validated SURGINT checkpoint:

```bash
SURGINT_CHECKPOINT=/path/to/checkpoint SURGINT_DEVICE=cpu python -m uvicorn services.api.app:app --host 127.0.0.1 --port 7860
```

Then open <http://127.0.0.1:7860>.

## Training and evaluation

```bash
# local training
python -m pipelines.train --config configs/train.yaml

# training tracked in W&B, with the best checkpoint logged as a candidate artifact
python -m pipelines.train --config configs/train.yaml --wandb-project YOUR_WANDB_PROJECT --wandb-entity YOUR_WANDB_ENTITY

# detection evaluation
python -m pipelines.evaluate outputs/runs/<run-id>/best --config configs/eval.yaml

# end-to-end detection, tracking, and inventory evaluation
python -m pipelines.evaluate outputs/runs/<run-id>/best --config configs/eval_sessions.yaml
```

Optionally log an already validated checkpoint to W&B as a candidate artifact:

```bash
python -m pipelines.register_model outputs/runs/<run-id>/best --project YOUR_WANDB_PROJECT --entity YOUR_WANDB_ENTITY
```

## Citation

The following BibTeX entry cites this project:

```bibtex
@software{ghita_2026_surgint_instruments,
  author = {Ahmed Ghita},
  title = {SURGINT Instruments},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/AhmedGhita1/surgint-instruments}
}
```

## License

The source code is released under the [MIT License](LICENSE). Model weights and datasets are separate artifacts and may have their own license terms.
