"""Register an existing validated checkpoint as a W&B model artifact."""

import argparse
import importlib
from pathlib import Path

from pipelines.wandb_tracking import DEFAULT_MODEL_ARTIFACT, log_candidate_model
from surgint.artifacts import ModelManifest


def register_model(
    checkpoint: Path,
    project: str,
    entity: str,
    artifact_name: str = DEFAULT_MODEL_ARTIFACT,
    alias: str = "candidate",
) -> str:
    """Upload one checkpoint and return its verified immutable W&B reference."""
    ModelManifest.load(checkpoint)

    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'W&B registration requires the training dependency: pip install -e ".[training]"'
        ) from error

    with wandb.init(
        project=project,
        entity=entity,
        job_type="model-bootstrap",
        name=f"register-{checkpoint.name}",
    ) as run:
        artifact = log_candidate_model(run, checkpoint, artifact_name, alias)
        reference = artifact.qualified_name

    if not reference:
        raise RuntimeError("W&B did not return an immutable artifact reference")

    wandb.Api().artifact(reference, type="model")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--artifact-name", default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--alias", default="candidate")
    args = parser.parse_args()

    reference = register_model(
        args.checkpoint,
        args.project,
        args.entity,
        args.artifact_name,
        args.alias,
    )
    print(f"registered {reference}")


if __name__ == "__main__":
    main()
