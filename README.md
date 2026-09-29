---
title: SURGINT Instruments
sdk: docker
app_port: 7860
---

# SURGINT Instruments

[![CI](https://github.com/AhmedGhita1/SURGINT/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedGhita1/SURGINT/actions/workflows/ci.yml)

*surgint-instruments is part of the surgical intelligence (SURGINT) family of projects.*

**SURGINT Instruments** is focused on surgical instruments inventory and handling procedures. It turns a surgical-tray video or live camera feed into an instrument inventory. It detects and tracks tray items using RT-DETR and ByteTracker, then applies explicit ontology-backed rules to produce handling procedure guidance.


> **Research demonstration only.** SURGINT Instruments has not been clinically validated, and must not be used for clinical decisions, patient care, or safety-critical instrument accounting.

![SURGINT Instruments architecture](docs/SURGINT-architecture-v2.png)

## Demo

The public GPU demo is available on [Hugging Face Spaces](https://huggingface.co/spaces/AhmedGhita/surgint).


## Run with Docker

The production image is GPU-only. A local deployment requires Docker with BuildKit, the NVIDIA Container Toolkit, an NVIDIA driver compatible with CUDA 12.1, and a W&B API key that can read the release artifact.

The following commands build the image from the immutable model artifact used by the hosted demo:

```bash
export WANDB_API_KEY="<your W&B API key>"

docker build --target production \
  --secret id=WANDB_API_KEY,env=WANDB_API_KEY \
  --build-arg WANDB_ARTIFACT=YOUR_WANDB_ENTITY/YOUR_WANDB_PROJECT/surgint-detector:v0 \
  -t surgint-instruments:1.0.0 .
```

The model is downloaded once during the image build and stored inside the final image. The W&B credential is exposed only as a build secret; it is not copied into the image.

```bash
docker run --rm --gpus all -p 7860:7860 surgint-instruments:1.0.0
```

The browser interface is then available at [http://localhost:7860](http://localhost:7860), and the generated OpenAPI documentation is available at [http://localhost:7860/docs](http://localhost:7860/docs).



## Development

SURGINT Instruments supports Python 3.10 and later. An editable development installation contains the model, serving, training, evaluation, rendering, and test dependencies:

```bash
python -m pip install -e ".[dev,render,serving,training]"
```

## Training and evaluation

Training and evaluation are offline workflows. They consume versioned datasets and produce reproducible run directories under `outputs/runs`; they do not depend on production traffic. Production observations may become a future data source only after a separate collection, curation, labeling, and dataset-versioning process.

The configuration files under `configs/` define the dataset, model revision, hyperparameters, splits, and metrics. Typical entry points are:

```bash
# Local training
python -m pipelines.train --config configs/train.yaml

# Training tracked in W&B, with the best checkpoint logged as a candidate artifact
python -m pipelines.train --config configs/train.yaml \
  --wandb-project YOUR_WANDB_PROJECT --wandb-entity YOUR_WANDB_ENTITY

# Detection evaluation
python -m pipelines.evaluate outputs/runs/<run-id>/best --config configs/eval.yaml

# End-to-end detection, tracking, and inventory evaluation
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
