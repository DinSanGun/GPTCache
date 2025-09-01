# tools/make_expensive_biased_dataset.py
import argparse, hashlib, random
from pathlib import Path

def is_expensive(p: str, frac: float) -> bool:
    h = int(hashlib.sha1(p.encode("utf-8")).hexdigest(), 16) % 100
    return h < int(frac * 100)

def build_tokenizer(tok_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    except Exception as e:
        print(f"[warn] transformers not available or tokenizer load failed: {e}")
        return None

def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    if tokenizer is None:
        # Fallback heuristic: cut by words; crude but avoids crashes
        return " ".join(text.split()[: max_tokens])
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return text
    trimmed = ids[:max_tokens]
    # decode back to text (drop special tokens if any)
    return tokenizer.decode(trimmed, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--n", type=int, default=20000, help="base prompts to draw before duplication")
    ap.add_argument("--hi-frac", type=float, default=0.20, help="fraction considered expensive")
    ap.add_argument("--dup", type=int, default=2, help="extra duplicates per expensive prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=480, help="truncate prompts to this many tokens")
    ap.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HF tokenizer name (bert-family for your ONNX model)")
    args = ap.parse_args()

    src = [ln.strip() for ln in Path(args.infile).read_text(encoding="utf-8").splitlines() if ln.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(src)
    base = src[:args.n]

    # classify & duplicate expensive prompts (to increase reuse under churn)
    expensive = [p for p in base if is_expensive(p, args.hi_frac)]
    duped = base + [p for p in expensive for _ in range(args.dup)]
    rng.shuffle(duped)

    tok = build_tokenizer(args.tokenizer)
    safe = [truncate_to_tokens(p, tok, args.max_tokens) for p in duped]

    # ensure non-empty after truncation
    safe = [s for s in safe if s.strip()]

    Path(args.outfile).write_text("\n".join(safe) + "\n", encoding="utf-8")
    print(f"wrote {len(safe)} prompts to {args.outfile} (base {len(base)}, expensive {len(expensive)}, dup x{args.dup}, max_tokens={args.max_tokens})")

if __name__ == "__main__":
    main()
