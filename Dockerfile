FROM python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin mancala
COPY deploy/requirements-container.txt /tmp/requirements-container.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements-container.txt
COPY ml /app/ml
COPY docs/ai-service /app/docs/ai-service
USER 10001:10001
EXPOSE 8080
STOPSIGNAL SIGTERM
CMD ["python", "-m", "uvicorn", "ml.alphazero_lite.service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-graceful-shutdown", "25"]
