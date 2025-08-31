import argparse, json, os
from pathlib import Path
from datasets import load_dataset

def _get_messages(record):
    """
    Return a Python list of OpenAI-style messages [{'role': 'user'|'assistant'|'system', 'content': str}, ...]
    from whatever field the dataset uses.
    """
    # Try common field names
    for k in ("conversation", "messages", "conversation_text", "conversation_json"):
        if k in record:
            conv = record[k]
            break
    else:
        # last resort: first JSON-looking string field
        conv = None
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
    if not isinstance(conv, list):
        return []
    return conv

def iter_user_prompts_english(streaming: bool, max_records: int | None):
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=streaming)
    n = 0
    for rec in ds:
        # Use dataset-provided language tag (per dataset card)
        if rec.get("language") != "English":
            continue

        msgs = _get_messages(rec)
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    yield content
                    n += 1
                    if max_records and n >= max_records:
                        return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lmsys_user_prompts_en.txt", help="Output text file (one prompt per line)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N prompts (0 = all)")
    ap.add_argument("--stream", action="store_true", help="Stream without loading into RAM (recommended)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for p in iter_user_prompts_english(streaming=args.stream, max_records=(args.limit or None)):
            f.write(p.replace("\n", " ").strip() + "\n")
            count += 1

    print(f"Wrote {count} English user prompts -> {out}")

if __name__ == "__main__":
    main()
