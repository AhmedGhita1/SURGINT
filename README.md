---
title: SURGINT
sdk: docker
app_port: 7860
---

# SURGINT

SURGINT (*Surgical Instrument Intelligence*) is a GPU-backed demonstration of surgical
instrument detection, tracking, inventory estimation, and rule-based handling guidance.

> **Research demonstration only.** SURGINT is not a medical device, has not been clinically
> validated, and must not be used for clinical decisions, patient care, or safety-critical
> instrument accounting.

The public demonstration runs at
[huggingface.co/spaces/AhmedGhita/surgint](https://huggingface.co/spaces/AhmedGhita/surgint).

## What the release contains

- RT-DETR detection for 13 tray-item classes.
- ByteTrack-style association and session-level inventory estimation.
- An OWL ontology and versioned demonstration policy for handling recommendations.
- A FastAPI service with a small browser interface.
- Complete-video upload with server-side 1 FPS sampling and four-frame GPU batches.
- Live-camera processing through the same session runtime.
- A CUDA-only Docker deployment for a Hugging Face GPU Space.
- Training and evaluation pipelines with W&B experiment and model-artifact integration.

The release model is baked into the production image from the immutable W&B reference
`SETLabs-HCT/surgint/surgint-detector:v0`. The model artifact includes its labels,
preprocessing contract, provenance, and weights digest.

## Runtime flow

```text
recorded video: one upload -> sample -> preprocess batches -> GPU inference
                                                -> chronological tracking
                                                -> inventory -> policy result

live camera:    frame stream -> GPU inference -> tracking -> inventory -> policy result
```

Recorded uploads are limited to 256 MiB and 300 sampled frames. At the default 1 FPS,
this represents at most five minutes of video. The service uses one GPU inference lock and
keeps at most eight active in-memory sessions.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Model readiness |
| `POST /v1/sessions` | Create isolated tracking state |
| `POST /v1/sessions/{id}/video` | Upload and process one complete recording |
| `POST /v1/sessions/{id}/frames` | Process one live-camera frame |
| `POST /v1/sessions/{id}/finalize` | Freeze inventory and resolve handling guidance |
| `DELETE /v1/sessions/{id}` | Release the session |

FastAPI exposes the generated API documentation at `/docs`.

## Evaluation status

The latest local synthetic tracking evaluation covers 4,200 frames from seven complete
sessions. It reports MOTA `0.554`, IDF1 `0.565`, class accuracy `0.943`, and inventory mean
absolute error `0.934` items per class. Class-level exact inventory agreement is `0.396`;
no evaluated session achieved an entirely exact inventory.

These results describe synthetic data and do not establish clinical performance. The
release prioritizes a reproducible end-to-end demonstration over production accuracy.

## Deployment

The production Docker build requires:

- Build variable `WANDB_ARTIFACT=SETLabs-HCT/surgint/surgint-detector:v0`
- Build secret `WANDB_API_KEY`
- NVIDIA GPU runtime compatible with CUDA 12.1

The image downloads the immutable artifact during its build. The credential does not enter
the final image. GitHub Actions tests the Python package, builds and smoke-tests the container,
and synchronizes `main` to the public Hugging Face Docker Space.

## Repository layout

```text
surgint/       reusable model, evaluation, tracking, inventory, ontology and policy code
services/api/  FastAPI application and browser interface
pipelines/     offline training, evaluation and W&B registration entry points
configs/       training and evaluation configurations
tests/         unit, integration and real-model checks
```

`pipelines/build_dataset.py` is intentionally still a stub. Dataset collection, curation,
labeling, and publication remain offline work and are not required to run this release.

## Known limitations

- The handling policy is explicitly demonstration-only.
- Uploaded-video processing is synchronous and returns after the complete job finishes.
- Sessions live only in one process and do not survive a container restart.
- The optimized multi-scale deformable-attention CUDA extension is not compiled; Transformers
  uses its PyTorch CUDA fallback.
- Model behavior has not been evaluated on clinical deployments or diverse real-world sites.
