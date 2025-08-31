import argparse, json
from pathlib import Path
from typing import Iterable, Optional
from datasets import load_dataset
from transformers import AutoTokenizer

def get_messages(record: dict) -> list[dict]:
    """
    Return a list of OpenAI-style messages: [{'role': 'user'|'assistant'|'system', 'content': str}, ...]
    Handles a few possible field names and JSON-string payloads.
    """
    conv = None
    for key in ("conversation", "messages", "conversation_text", "conversation_json"):
        if key in record:
            conv = record[key]
            break
    if conv is None:
        # last resort: first JSON-looking string field
        for v in record.values():
            if isinstance(v, str) and v.strip().startswith("["):
                conv = v
                break
    if conv is None:
        return []
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:
            return []
    if isinstance(conv, dict) and "messages" in conv:
        conv = conv["messages"]
    return conv if isinstance(conv, list) else []

def iter_user_prompts_en(streaming: bool) -> Iterable[str]:
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=streaming)
    for rec in ds:
        if rec.get("language") != "English":
            continue
        msgs = get_messages(rec)
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    yield content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lmsys_user_prompts_en_filtered.txt",
                    help="Output text file (one prompt per line)")
    ap.add_argument("--target-count", type=int, default=200_000,
                    help="How many prompts to keep (stop when reached)")
    ap.add_argument("--max-tokens", type=int, default=480,
                    help="Discard (or truncate) prompts with >= this many tokens. Keep a margin below 512.")
    ap.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HF tokenizer id to estimate token length (choose the one matching your ONNX embedder)")
    ap.add_argument("--truncate", action="store_true",
                    help="If set, truncate long prompts to max-tokens instead of skipping them.")
    ap.add_argument("--stream", action="store_true", default=True,
                    help="Use streaming mode to avoid loading all data into RAM.")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    kept = 0
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for text in iter_user_prompts_en(streaming=args.stream):
            # +2 gives room for special tokens ([CLS], [SEP]) depending on tokenizer
            ids = tok.encode(text, add_special_tokens=True, truncation=False)
            if len(ids) > args.max_tokens:
                if args.truncate:
                    # Truncate safely with tokenizer to preserve proper token boundaries
                    enc = tok(text, add_special_tokens=True, truncation=True, max_length=args.max_tokens)
                    text = tok.decode(enc["input_ids"], skip_special_tokens=True)
                else:
                    continue  # skip this prompt entirely

            # Single-line, no newlines
            f.write(text.replace("\n", " ").strip() + "\n")
            kept += 1
            written += 1
            if kept >= args.target_count:
                break

    print(f"Wrote {written} prompts to {out} (<= {args.max_tokens} tokens, tokenizer='{args.tokenizer}')")

if __name__ == "__main__":
    main()
