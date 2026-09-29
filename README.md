[![CI](https://github.com/AhmedGhita1/SURGINT/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedGhita1/SURGINT/actions/workflows/ci.yml)
[![Python 3.10 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/AhmedGhita/surgint)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/AhmedGhita/surgint-instruments-rt-detr)

# SURGINT Instruments



*surgint-instruments is part of the surgical intelligence (SURGINT) family of projects.*

**SURGINT Instruments** is focused on surgical instruments inventory and handling procedures. It turns a surgical-tray video or live camera feed into an instrument inventory. It detects and tracks tray items using RT-DETR and ByteTracker, then applies explicit ontology-backed rules to produce handling procedure guidance.


> **Research demonstration only.** SURGINT Instruments has not been clinically validated, and must not be used for clinical decisions, patient care, or safety-critical instrument accounting.

![SURGINT Instruments architecture](docs/SURGINT-architecture-v2.svg)

## Run with Docker

Running the Docker image requires Docker, the NVIDIA Container Toolkit, and an NVIDIA driver
compatible with CUDA 12.1.

```bash
docker pull ghcr.io/ahmedghita1/surgint-instruments:1.0.0
docker run --rm --gpus all -p 7860:7860 \
  ghcr.io/ahmedghita1/surgint-instruments:1.0.0
```

Open <http://localhost:7860> after the container reports healthy.



## Development

SURGINT Instruments supports Python 3.10 and 3.12. An editable development installation contains the model, serving, training, evaluation, rendering, and test dependencies:

```bash
python -m pip install -e ".[dev,render,serving,training]"
```

Run the API locally with a validated SURGINT checkpoint:

```bash
SURGINT_CHECKPOINT=/path/to/checkpoint SURGINT_DEVICE=cpu \
  python -m uvicorn services.api.app:app --host 127.0.0.1 --port 7860
```

Then open <http://127.0.0.1:7860>.

## Training and evaluation

Training and evaluation are offline workflows. They consume versioned datasets and produce reproducible run directories under `outputs/runs`.

Typical entry points:

```bash
# local training
python -m pipelines.train --config configs/train.yaml

# training tracked in W&B, with the best checkpoint logged as a candidate artifact
python -m pipelines.train --config configs/train.yaml \
  --wandb-project YOUR_WANDB_PROJECT --wandb-entity YOUR_WANDB_ENTITY

# detection evaluation
python -m pipelines.evaluate outputs/runs/<run-id>/best --config configs/eval.yaml

# end-to-end detection, tracking, and inventory evaluation
python -m pipelines.evaluate outputs/runs/<run-id>/best --config configs/eval_sessions.yaml
```

An already validated checkpoint can be registered separately:

```bash
python -m pipelines.register_model outputs/runs/<run-id>/best \
  --project YOUR_WANDB_PROJECT --entity YOUR_WANDB_ENTITY
```

## Citation

The following BibTeX entry cites this project:

```bibtex
@software{ghita_2026_surgint_instruments,
  author = {Ahmed Ghita},
  title = {SURGINT Instruments},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/AhmedGhita1/SURGINT}
}
```

## License

The source code is released under the [MIT License](LICENSE). Model weights and datasets are separate artifacts and may have their own license terms.
