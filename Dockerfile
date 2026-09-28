FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime@sha256:ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SURGINT_CHECKPOINT=/app/model \
    SURGINT_DEVICE=cuda \
    SURGINT_MAX_SESSIONS=8

RUN useradd --create-home --uid 10001 surgint

# Install the serving package before copying the model so a model promotion
# does not invalidate the dependency layer.
COPY pyproject.toml README.md /src/
COPY surgint/ /src/surgint/
COPY services/ /src/services/
RUN python -m pip install --retries 10 --timeout 120 "/src[serving]" \
    && rm -rf /src

WORKDIR /app

ARG MODEL_PATH=models/fair-raven
COPY --chown=surgint:surgint ${MODEL_PATH}/ /app/model/

USER surgint

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health/ready', timeout=2)"]

CMD ["python", "-m", "uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
