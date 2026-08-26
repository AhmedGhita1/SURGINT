# SURGINT

*Real-Time Surgical Instrument Detection, Tracking & Inventory*

A real-time vision system that detects and tracks surgical instruments on a static tray from a moving camera and maintains a stateful instrument inventory.

## Detection workflow

Run from the repository root with the project installed in the active environment.

```bash
python -m pytest -q
python scripts/train.py
python scripts/evaluate.py outputs/runs/<train-run>/best
python scripts/infer.py <image> --checkpoint outputs/runs/<train-run>/best
```

Training and evaluation create separate directories under `outputs/runs/`. Each run contains its root `config.yaml`, dataset and model `manifest.yaml`, JSON Lines `run.log`, and final `summary.json`. Training runs additionally contain validation-selected `best/` and resumable `latest/` checkpoints.
