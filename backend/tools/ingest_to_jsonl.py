#!/usr/bin/env python3
"""
ingest_to_jsonl.py  →  robust converter to our JSONL schema:
  {"id": "...", "ref": "...", "text": "...", "source": "bible|quran|talmud|..."}
It auto-detects common formats:
  1) JSONL passthrough (remap keys)
  2) USFM (\c, \v) parsing
  3) Plain "Book 1:2 Text..." lines (several regex variants)
  4) Fallback: paragraph chunks with synthetic refs

Usage:
  python ingest_to_jsonl.py <input_file> <output_jsonl> --source bible
"""
import argparse, json, os, re, sys
from typing import Iterator, Dict, List

# ----- helpers ---------------------------------------------------------------

def write_jsonl(objs: Iterator[Dict], out_path: str) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for obj in objs:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n

def norm_book(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())

# ----- detectors & parsers ---------------------------------------------------

def try_jsonl(inp: str, source: str) -> Iterator[Dict]:
    """Accept jsonl already; map common key variants."""
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        try:
            test = json.loads(first)
        except Exception:
            return iter(())  # not jsonl
    # reopen to stream
    def gen():
        with open(inp, "r", encoding="utf-8", errors="ignore") as f2:
            for line in f2:
                if not line.strip(): continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if not text: continue
                ref = obj.get("ref") or obj.get("reference") or obj.get("verse_ref") or ""
                _id = obj.get("id") or obj.get("verse_id") or ref or str(hash(text))
                yield {"id": str(_id), "ref": str(ref), "text": text.strip(), "source": source}
    return gen()

def try_usfm(inp: str, source: str) -> Iterator[Dict]:
    """Parse USFM with \c and \v markers."""
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4000)
    if "\\c " not in sample or "\\v " not in sample:
        return iter(())  # not USFM

    def gen():
        book = "Book"
        chap = "1"
        with open(inp, "r", encoding="utf-8", errors="ignore") as f2:
            buf = []
            verse = None
            for raw in f2:
                ln = raw.rstrip("\n")
                if ln.startswith("\\id "):
                    # \id GEN ...
                    parts = ln.split(maxsplit=2)
                    if len(parts) >= 2:
                        book = parts[1]
                elif ln.startswith("\\h "):
                    book = norm_book(ln[3:])
                elif ln.startswith("\\c "):
                    chap = ln[3:].strip().split()[0]
                elif ln.startswith("\\v "):
                    # flush previous
                    if verse is not None:
                        text = " ".join(buf).strip()
                        if text:
                            ref = f"{book} {chap}:{verse}"
                            vid = f"{source}-{book}-{chap}-{verse}".replace(" ", "_")
                            yield {"id": vid, "ref": ref, "text": text, "source": source}
                    # start new
                    bits = ln[3:].split(maxsplit=1)
                    verse = bits[0]
                    buf = [bits[1]] if len(bits) > 1 else []
                else:
                    if verse is not None:
                        buf.append(ln.strip())
            # tail
            if verse is not None:
                text = " ".join(buf).strip()
                if text:
                    ref = f"{book} {chap}:{verse}"
                    vid = f"{source}-{book}-{chap}-{verse}".replace(" ", "_")
                    yield {"id": vid, "ref": ref, "text": text, "source": source}
    return gen()

PLAIN_PATTERNS = [
    # Genesis 1:1 In the beginning...
    re.compile(r'^([A-Za-z][A-Za-z0-9 .\-]*)\s+(\d+):(\d+)\s+(.+)$'),
    # 1 Samuel 3:10 ...
    re.compile(r'^(\d+\s+[A-Za-z][A-Za-z0-9 .\-]*)\s+(\d+):(\d+)\s+(.+)$'),
    # Phil 4:6-7 short books
    re.compile(r'^([A-Za-z][A-Za-z0-9 .\-]+)\s+(\d+):(\d+(?:-\d+)?)\s+(.+)$'),
]

def try_plain_verse_lines(inp: str, source: str) -> Iterator[Dict]:
    """Lines like: 'Genesis 1:1 Text...' or '1 Samuel 3:10 Text...'"""
    # quick sniff
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        sample = "".join([next(f, "") for _ in range(40)])
    if not any(p.search(sample) for p in PLAIN_PATTERNS):
        return iter(())  # not this format

    def gen():
        with open(inp, "r", encoding="utf-8", errors="ignore") as f2:
            for line in f2:
                s = line.strip()
                if not s: continue
                m = None
                for pat in PLAIN_PATTERNS:
                    m = pat.match(s)
                    if m: break
                if not m: continue
                book, chap, verse, txt = m.groups()
                book = norm_book(book); chap = str(chap); verse = str(verse)
                ref = f"{book} {chap}:{verse}"
                vid = f"{source}-{book.replace(' ','_')}-{chap}-{verse}"
                yield {"id": vid, "ref": ref, "text": txt.strip(), "source": source}
    return gen()

def fallback_paragraph_chunks(inp: str, source: str, chunk_chars=600) -> Iterator[Dict]:
    """Last resort: chunk by paragraphs into pseudo-refs."""
    def gen():
        with open(inp, "r", encoding="utf-8", errors="ignore") as f:
            buf: List[str] = []
            idx = 1
            for line in f:
                t = line.strip()
                if not t:
                    if buf:
                        txt = " ".join(buf).strip()
                        ref = f"{source}-p{idx}"
                        vid = f"{source}-p{idx}"
                        yield {"id": vid, "ref": ref, "text": txt, "source": source}
                        buf, idx = [], idx + 1
                else:
                    buf.append(t)
                    # flush long chunks
                    if sum(len(x)+1 for x in buf) >= chunk_chars:
                        txt = " ".join(buf).strip()
                        ref = f"{source}-p{idx}"
                        vid = f"{source}-p{idx}"
                        yield {"id": vid, "ref": ref, "text": txt, "source": source}
                        buf, idx = [], idx + 1
            if buf:
                txt = " ".join(buf).strip()
                ref = f"{source}-p{idx}"
                vid = f"{source}-p{idx}"
                yield {"id": vid, "ref": ref, "text": txt, "source": source}
    return gen()

# ----- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--source", default="bible")
    args = ap.parse_args()

    # Try in order
    parsers = [
        try_jsonl,
        try_usfm,
        try_plain_verse_lines,
    ]
    for p in parsers:
        it = p(args.input, args.source)
        # Peek
        head = []
        for i, row in enumerate(it):
            head.append(row)
            if i >= 9: break
        if head:
            # re-yield head then the rest by re-running parser
            def again():
                for r in head: yield r
                # re-run to stream the rest
                for r in p(args.input, args.source):
                    pass  # consumed; avoid duplication
            # We need to stream full generator; rebuild a single pass:
            # Simpler: run parser again and write directly
            n = write_jsonl(iter(head), args.output)  # write head first
            # now write the rest skipping first len(head) rows
            count = 0
            for j, r in enumerate(p(args.input, args.source)):
                if j < len(head): continue
                with open(args.output, "a", encoding="utf-8") as w:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
            print(f"[ingest] wrote {n+count} rows to {args.output}")
            return

    # Fallback: paragraph chunks
    it = fallback_paragraph_chunks(args.input, args.source)
    n = write_jsonl(it, args.output)
    print(f"[ingest:fallback] wrote {n} paragraph chunks to {args.output}")

if __name__ == "__main__":
    main()
