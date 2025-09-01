import argparse, hashlib
from pathlib import Path
import random

def is_expensive(p: str, frac: float) -> bool:
    h = int(hashlib.sha1(p.encode("utf-8")).hexdigest(), 16) % 100
    return h < int(frac * 100)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--n", type=int, default=20000, help="total prompts to emit (before duplication)")
    ap.add_argument("--hi-frac", type=float, default=0.20, help="target fraction considered expensive")
    ap.add_argument("--dup", type=int, default=2, help="how many extra duplicates for each expensive prompt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = [ln.strip() for ln in Path(args.infile).read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(args.seed).shuffle(src)
    base = src[:args.n]

    expensive = [p for p in base if is_expensive(p, args.hi_frac)]
    duped = base + [p for p in expensive for _ in range(args.dup)]
    random.Random(args.seed).shuffle(duped)

    Path(args.outfile).write_text("\n".join(duped) + "\n", encoding="utf-8")
    print(f"wrote {len(duped)} prompts to {args.outfile} (base {len(base)}, expensive {len(expensive)}, dup x{args.dup})")

if __name__ == "__main__":
    main()
