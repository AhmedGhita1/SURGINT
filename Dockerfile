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


# Release models are public. The full Hub commit SHA makes this input immutable.
FROM python:3.12.14-slim-bookworm AS release-model

ARG HF_MODEL_ID
ARG HF_MODEL_REVISION

RUN python -m pip install --no-cache-dir huggingface-hub==0.36.0
RUN test -n "${HF_MODEL_ID}" \
    && test -n "${HF_MODEL_REVISION}" \
    && python -c "import os, re; from huggingface_hub import snapshot_download; revision = os.environ['HF_MODEL_REVISION']; assert re.fullmatch(r'[0-9a-f]{40}', revision), 'HF_MODEL_REVISION must be a full commit SHA'; snapshot_download(repo_id=os.environ['HF_MODEL_ID'], revision=revision, local_dir='/model', allow_patterns=['config.json', 'model.safetensors', 'manifest.yaml'])" \
    && rm -rf /model/.cache


FROM runtime AS production
COPY --from=release-model --chown=surgint:surgint /model/ /app/model/
RUN python -c "from surgint.artifacts import ModelManifest; ModelManifest.load('/app/model')"
