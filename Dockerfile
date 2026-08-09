# CyberSentinel application image (API + Streamlit UI).
#
# Training is deliberately out of scope for this image: it needs a GPU and a
# multi-gigabyte CUDA stack, and the application must remain runnable without
# retraining. Train on the host with scripts/train.py and mount the adapter in.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is used by the container health checks below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer: copied first so code edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[ui,eval]"

COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY data/knowledge_base/ ./data/knowledge_base/

# Run as an unprivileged user: this service parses hostile input by design.
RUN useradd --create-home --uid 10001 cybersentinel \
    && mkdir -p /app/data/processed /app/models \
    && chown -R cybersentinel:cybersentinel /app
USER cybersentinel

EXPOSE 8000 8501


FROM base AS api
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "cybersentinel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS ui
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
