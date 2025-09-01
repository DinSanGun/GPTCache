# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (Optional) system deps; keep minimal. Uncomment if you need to compile extras.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt requirements-real.txt /app/
RUN python -m pip install -U pip && \
    pip install -r requirements.txt

# Build-arg to optionally install real/semantic extras (onnxruntime, etc.)
ARG INSTALL_REAL_EXTRAS=0
RUN if [ "$INSTALL_REAL_EXTRAS" = "1" ]; then \
      pip install -r requirements-real.txt; \
    fi

# Bring the rest of the project in
COPY . /app

# Default entrypoint just shows the bench CLI help;
# override CMD in `docker run` for specific experiments.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["python", "tools/bench_policies.py", "--help"]
