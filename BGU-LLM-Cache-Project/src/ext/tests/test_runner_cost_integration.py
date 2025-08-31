import sys, subprocess, csv
from pathlib import Path
import pytest

@pytest.mark.slow
def test_runner_cost_aware_e2e(tmp_path: Path):
    # Paths under tmp so we don't touch repo dirs
    artifacts = tmp_path / "artifacts"
    out_dir = tmp_path / "out"
    cfg = tmp_path / "cfg.yaml"

    # Minimal config that triggers some cache activity
    cfg.write_text(f"""
mode: mock
cache:
  type: semantic
  eviction: COST_AWARE
  max_size: 4
  clean_size: 1
  similarity: distance
  similarity_threshold: 0.98
  cost_metric: latency_ms
  cost_decay: 0.0
paths:
  artifacts_dir: {artifacts.as_posix()}
  out_dir: {out_dir.as_posix()}
  reset_artifacts: true
run:
  prompts: 12
  warm_repeats: 1
  shuffle_warm: true
  strict_cold: true
""", encoding="utf-8")

    # Project root = BGU-LLM-Cache-Project
    ROOT = Path(__file__).resolve().parents[3]

    # Execute the bench
    proc = subprocess.run(
        [sys.executable, "-m", "src.bench.runner", "--config", str(cfg)],
        cwd=ROOT, text=True, capture_output=True
    )
    assert proc.returncode == 0, f"runner failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # Assert outputs exist
    summary_path = out_dir / "summary.csv"
    detail_path = out_dir / "detail.csv"
    assert summary_path.is_file(), "summary.csv not written"
    assert detail_path.is_file(), "detail.csv not written"

    # Quick, dependency-light checks (csv module instead of pandas)
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # numeric sanity
        assert "cost_saved_total" in row
        saved = float(row["cost_saved_total"])
        assert saved >= 0.0
        hit_warm = float(row["hit_rate_warm"])
        assert 0.0 <= hit_warm <= 1.0

    with detail_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # we should have both cold and warm rows
        phases = {r["phase"].split(" ")[0] for r in reader}
        assert any(p.startswith("cold") for p in phases)
        assert any(p.startswith("warm") for p in phases)
