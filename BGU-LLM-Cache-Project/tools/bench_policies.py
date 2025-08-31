# tools/bench_policies.py
import sys, subprocess, argparse, json, time
from pathlib import Path
import yaml

DEFAULT_POLICIES = ["FIFO", "LFU", "COST_AWARE"]

def write_knobs(knobs_path: Path, dataset_path: Path, out_dir: Path, artifacts_dir: Path,
                shuffle_warm: bool, strict_cold: bool):
    knobs = {
        "paths": {
            "out_dir": out_dir.as_posix(),
            "artifacts_dir": artifacts_dir.as_posix(),
            "reset_artifacts": True,
        },
        "run": {
            "prompts_file": dataset_path.as_posix(),
            "shuffle_warm": bool(shuffle_warm),
            "strict_cold": bool(strict_cold),
        },
    }
    knobs_path.parent.mkdir(parents=True, exist_ok=True)
    knobs_path.write_text(yaml.safe_dump(knobs, sort_keys=False), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock","ollama"], required=True)
    ap.add_argument("--base-config", required=True, help="Base YAML (mock or ollama)")
    ap.add_argument("--dataset", required=True, help="TXT: one prompt per line")
    ap.add_argument("--out-root", default="results/suite", help="Parent folder for runs")
    ap.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    ap.add_argument("--prompts", type=int, default=100, help="Number of prompts to use")
    ap.add_argument("--sim-thr", type=float, default=0.98)
    ap.add_argument("--clean-size", type=int, default=1)
    ap.add_argument("--max-size", type=int, default=4)
    ap.add_argument("--warm-repeats", type=int, default=1)
    ap.add_argument("--shuffle-warm", action="store_true", default=True)
    ap.add_argument("--strict-cold", action="store_true", default=True)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]  # repo root of bench project
    base_cfg = (ROOT / args.base_config).resolve()
    dataset = (ROOT / args.dataset).resolve()
    out_root = (ROOT / args.out_root / args.mode).resolve()

    assert base_cfg.is_file(), f"base config not found: {base_cfg}"
    assert dataset.is_file(), f"dataset not found: {dataset}"
    out_root.mkdir(parents=True, exist_ok=True)

    all_runs = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    for pol in args.policies:
        run_dir = out_root / f"{pol.lower()}_{ts}"
        art_dir = run_dir / "artifacts"
        knobs_path = run_dir / "knobs.yaml"
        write_knobs(knobs_path, dataset, run_dir, art_dir, args.shuffle_warm, args.strict_cold)

        cmd = [
            sys.executable, "-m", "src.bench.runner",
            "--config", str(base_cfg),
            "--knobs", str(knobs_path),
            "--eviction", pol,
            "--prompts", str(args.prompts),
            "--sim-thr", str(args.sim_thr),
            "--max-size", str(args.max_size),
            "--clean-size", str(args.clean_size),
            "--warm-repeats", str(args.warm_repeats),
        ]
        print("[run]", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"runner failed for {pol}")

        all_runs.append({"policy": pol, "run_dir": run_dir.as_posix()})

    print(json.dumps(all_runs, indent=2))

if __name__ == "__main__":
    main()
