#!/usr/bin/env bash
set -euo pipefail
python -m src.bench.runner --config configs/mock.yaml
