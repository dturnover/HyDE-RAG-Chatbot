# chunker.py - split large texts into JSONL rows with stable ids/refs
import sys, json, os, re, textwrap
from typing import Iterator, Dict

MAX_CHARS = 1500  # safety cap per chunk

def yield_paragraphs(text: str) -> Iterator[str]:
    for block in re.split(r"\n\s*\n", text):
        t = block.strip()
        if not t:
            continue
        if len(t) <= MAX_CHARS:
            yield t
        else:
            # split long blocks by sentence-ish boundaries
            parts = re.split(r'(?<=[.!?])\s+', t)
            buf = ""
            for p in parts:
                if len(buf) + 1 + len(p) <= MAX_CHARS:
                    buf = (buf + " " + p).strip()
                else:
                    if buf:
                        yield buf
                    # if a single sentence is still huge, hard-wrap it
                    for w in textwrap.wrap(p, MAX_CHARS):
                        yield w
                    buf = ""
            if buf:
                yield buf

def main(trad: str, inp: str, outp: str):
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    out = open(outp, "w", encoding="utf-8")
    count = 0
    for i, para in enumerate(yield_paragraphs(raw), 1):
        obj: Dict = {
            "id": f"{trad}-p{i}",
            "ref": f"{trad}-p{i}",
            "text": para,
            "source": trad
        }
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        count += 1
    out.close()
    print(f"[chunker] wrote {count} chunks -> {outp}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python tools/chunker.py <trad> <in.txt> <out.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
