"""W&B integration owned by offline pipelines rather than the Surgint core."""

import importlib
from pathlib import Path
from typing import Any, Mapping

from surgint.artifacts import ModelManifest

DEFAULT_MODEL_ARTIFACT = "surgint-detector"


def init_training_run(
    project: str,
    entity: str | None,
    run_name: str,
    config: Mapping[str, Any],
):
    """Create the W&B run that records one training execution."""
    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'W&B tracking requires the training dependency: pip install -e ".[training]"'
        ) from error

    return wandb.init(
        project=project,
        entity=entity,
        job_type="train",
        name=run_name,
        config=dict(config),
    )


def log_epoch(run, record: Mapping[str, Any], epoch: int) -> None:
    """Record one completed training epoch."""
    run.log(dict(record), step=epoch)


def log_candidate_model(run, checkpoint: str | Path, name: str, run_alias: str):
    """Log a validated checkpoint as a candidate model artifact, without promoting it."""
    checkpoint = Path(checkpoint)
    ModelManifest.load(checkpoint)
    artifact = run.log_artifact(
        str(checkpoint),
        name=name,
        type="model",
        aliases=[run_alias],
    )
    artifact.wait()
    return artifact
