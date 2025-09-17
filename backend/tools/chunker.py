# tools/chunker.py
# Split various scripture sources into JSONL rows with stable, human refs.
import sys, json, re, argparse
from typing import Iterator, Dict

def write(out, row: Dict):
    out.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------- Parsers ----------

# Quran: "S|V|text" (Yusuf Ali pipe format)
_quran_line = re.compile(r'^\s*(\d+)\|(\d+)\|(.*)$')

def parse_quran(inp: str, source: str, out):
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _quran_line.match(line.strip())
            if not m: 
                continue
            s, v, txt = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            if not txt: 
                continue
            row = {
                "id": f"{source}-{s}-{v}",
                "source": source,             # "quran"
                "ref": f"{s}:{v}",            # e.g. "2:255"
                "chapter": s, "verse": v,
                "text": txt
            }
            write(out, row)

# Bible-like: book headings + {C:V} markers (ASV/NRSV)
_book_hdr1 = re.compile(r'^\s*===\s*([A-Za-z0-9 .:-]+?)\s*===\s*$')     # e.g. "=== Genesis ==="
_book_hdr2 = re.compile(r'^\s*([1-3]?\s*[A-Za-z][A-Za-z .:-]+)\.\s*$')  # e.g. "Genesis."
_cv_marker = re.compile(r'\{(\d+):(\d+)\}\s*')

def parse_bible_markers(inp: str, source: str, out):
    cur_book = ""
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line: 
                continue

            m1 = _book_hdr1.match(line)
            m2 = _book_hdr2.match(line)
            if m1: 
                cur_book = m1.group(1).strip()
                continue
            if m2 and (len(m2.group(1).split()) <= 3):  # short line with a book name + dot
                cur_book = m2.group(1).strip().replace(" :", ":")
                continue

            if not cur_book:
                # skip front matter / copyright pages
                continue

            # split by {C:V} markers and emit one verse per marker
            parts = _cv_marker.split(line)
            # parts pattern: [pre, chap, verse, after, chap, verse, after, ...]
            if len(parts) < 4:
                continue
            pre = parts[0]  # ignore text before first marker
            rest = parts[1:]
            for i in range(0, len(rest), 3):
                if i + 2 >= len(rest): 
                    break
                chap = int(rest[i]); verse = int(rest[i+1]); tail = rest[i+2].strip()
                # Tail may still include next {C:V}, but we emit per marker anyway.
                # Trim at next marker if present:
                nxt = _cv_marker.search(tail)
                verse_txt = tail[:nxt.start()].strip() if nxt else tail
                if not verse_txt: 
                    continue
                row = {
                    "id": f"{source}-{cur_book}-{chap}-{verse}",
                    "source": source,                      # "bible_asv" or "bible_nrsv"
                    "ref": f"{cur_book} {chap}:{verse}",   # e.g. "Genesis 1:1"
                    "book": cur_book, "chapter": chap, "verse": verse,
                    "text": verse_txt
                }
                write(out, row)

# Tanakh (JPS1917): lines like "9,6" then text; book from headings
_tanakh_book = re.compile(r'^\s*([1-3]?\s*[A-Za-z][A-Za-z .:-]+)\s*$')
_tanakh_cv   = re.compile(r'^\s*(\d+)[,.:\-]\s*(\d+)\s*$')

def parse_tanakh(inp: str, source: str, out):
    cur_book = ""
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Simple book line (pages list headings are usually lone book names)
            if _tanakh_book.match(line) and len(line.split()) <= 3:
                cur_book = line.strip().replace(" :", ":")
                continue
            # Lines like "9,6" on their own line precede the verse on the next line
            m = _tanakh_cv.match(line)
            if m:
                chap, verse = int(m.group(1)), int(m.group(2))
                # read next non-empty line(s) as text (until a blank or another marker)
                verse_lines = []
                for nxt in f:
                    s = nxt.strip()
                    if not s or _tanakh_cv.match(s):
                        # push file pointer back one line when we hit another cv marker
                        if _tanakh_cv.match(s):
                            # crude: store position not available; instead, process this line immediately
                            # by simulating a 'put-back' using a small buffer
                            # We can't un-read, so we process the found marker inline:
                            # emit current assembled verse, then continue with new marker
                            break
                        else:
                            break
                    verse_lines.append(s)
                txt = " ".join(verse_lines).strip()
                if cur_book and txt:
                    row = {
                        "id": f"{source}-{cur_book}-{chap}-{verse}",
                        "source": source,                  # "tanakh"
                        "ref": f"{cur_book} {chap}:{verse}",
                        "book": cur_book, "chapter": chap, "verse": verse,
                        "text": txt
                    }
                    write(out, row)
                continue

# Dhammapada: "number  text" (or any line starting with number)
_dham_num = re.compile(r'^\s*(\d+)[).:\-]\s*(.*)$')

def parse_dhammapada(inp: str, source: str, out):
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _dham_num.match(line.strip())
            if not m:
                continue
            n = int(m.group(1))
            txt = m.group(2).strip()
            if not txt:
                continue
            row = {
                "id": f"{source}-{n}",
                "source": source,                # "dhammapada"
                "ref": f"Dhammapada {n}",
                "text": txt
            }
            write(out, row)

# Bhagavad Gita:
# Preferred: "C|V|text" (pipe format). If you only have prose with commentary, use a cleaner source if you can.
_gita_pipe = re.compile(r'^\s*(\d+)\|(\d+)\|(.*)$')

def parse_gita(inp: str, source: str, out):
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _gita_pipe.match(line.strip())
            if not m:
                continue
            c, v, txt = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            if not txt:
                continue
            row = {
                "id": f"{source}-{c}-{v}",
                "source": source,                  # "gita"
                "ref": f"Bhagavad Gita {c}:{v}",
                "chapter": c, "verse": v,
                "text": txt
            }
            write(out, row)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Chunk scriptures to JSONL with clean refs")
    ap.add_argument("source", help="quran | bible_asv | bible_nrsv | tanakh | gita | dhammapada")
    ap.add_argument("inp", help="input .txt")
    ap.add_argument("out", help="output .jsonl (raw, without embeddings)")
    args = ap.parse_args()

    with open(args.out, "w", encoding="utf-8") as out:
        if args.source == "quran":
            parse_quran(args.inp, "quran", out)
        elif args.source in ("bible_asv", "bible_nrsv"):
            parse_bible_markers(args.inp, args.source, out)
        elif args.source == "tanakh":
            parse_tanakh(args.inp, "tanakh", out)
        elif args.source == "gita":
            parse_gita(args.inp, "gita", out)
        elif args.source == "dhammapada":
            parse_dhammapada(args.inp, "dhammapada", out)
        else:
            print("unknown source:", args.source)
            sys.exit(2)

if __name__ == "__main__":
    main()
