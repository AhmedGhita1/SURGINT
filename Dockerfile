FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime@sha256:ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SURGINT_CHECKPOINT=/app/model \
    SURGINT_DEVICE=cuda \
    SURGINT_MAX_SESSIONS=8 \
    SURGINT_MAX_VIDEO_BYTES=268435456 \
    SURGINT_MAX_VIDEO_FRAMES=300 \
    SURGINT_VIDEO_BATCH_SIZE=4 \
    SURGINT_VIDEO_SAMPLE_FPS=1.0 \
    SURGINT_NMS_IOU=0.7

RUN useradd --create-home --uid 1000 surgint

# Install the serving package before copying the model so a model promotion
# does not invalidate the dependency layer.
COPY pyproject.toml README.md LICENSE /src/
COPY surgint/ /src/surgint/
COPY services/ /src/services/
RUN python -m pip install --retries 10 --timeout 120 "/src[serving]" \
    && rm -rf /src

WORKDIR /app

USER surgint

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health/ready', timeout=2)"]

CMD ["python", "-m", "uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]


# Pull requests use a generated checkpoint and require no deployment secret.
FROM scratch AS ci-model
COPY models/ci/ /model/

FROM runtime AS ci
COPY --from=ci-model --chown=surgint:surgint /model/ /app/model/
RUN python -c "from surgint.artifacts import ModelManifest; ModelManifest.load('/app/model')"


# GitHub Actions provides WANDB_API_KEY as a BuildKit secret. Only the downloaded
# checkpoint crosses into the production image; W&B and its credential do not.
FROM python:3.12.14-slim-bookworm AS registry-model

ARG WANDB_ARTIFACT

RUN python -m pip install --no-cache-dir wandb==0.30.0
RUN --mount=type=secret,id=WANDB_API_KEY,mode=0444,required=true \
    test -n "${WANDB_ARTIFACT}" \
    && WANDB_API_KEY="$(cat /run/secrets/WANDB_API_KEY)" \
       WANDB_ARTIFACT="${WANDB_ARTIFACT}" \
       python -c "import os, wandb; wandb.Api().artifact(os.environ['WANDB_ARTIFACT']).download(root='/model')"


FROM runtime AS production
COPY --from=registry-model --chown=surgint:surgint /model/ /app/model/
RUN python -c "from surgint.artifacts import ModelManifest; ModelManifest.load('/app/model')"
