# corpus/combine_txts.py
import os, sys, pathlib, re

# Minimal, human-readable book order (prefix match against filenames)
BOOK_ORDER = [
  "Gen","Exod","Lev","Num","Deut","Josh","Judg","Ruth",
  "1Sam","2Sam","1Kings","2Kings","1Chron","2Chron","Ezra","Neh","Esth",
  "Job","Ps","Prov","Eccl","Song","Isa","Jer","Lam","Ezek","Dan",
  "Hos","Joel","Amos","Obad","Jonah","Mic","Nah","Hab","Zeph","Hag","Zech","Mal",
  "Matt","Mark","Luke","John","Acts","Rom","1Cor","2Cor","Gal","Eph","Phil","Col",
  "1Thes","2Thes","1Tim","2Tim","Titus","Philem","Heb","James","1Pet","2Pet",
  "1John","2John","3John","Jude","Rev"
]

def rank(path: str):
    name = pathlib.Path(path).stem
    for i, pref in enumerate(BOOK_ORDER):
        if name.lower().startswith(pref.lower()):
            return (0, i, name)
    # fallback: natural sort by name
    parts = re.split(r'(\d+)', name)
    parts = [int(p) if p.isdigit() else p.lower() for p in parts]
    return (1, *parts)

def main(src_dir: str, out_txt: str):
    p = pathlib.Path(src_dir)
    files = [str(x) for x in p.glob("*.txt")]
    if not files:
        print(f"[err] no .txt files in {src_dir}")
        sys.exit(2)
    files.sort(key=rank)

    out_chunks = []
    for fp in files:
        name = pathlib.Path(fp).stem
        print("appending", fp)
        txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
        txt = txt.replace("\r\n","\n").replace("\r","\n").strip()
        out_chunks.append(f"\n\n=== {name} ===\n{txt}\n")

    joined = "".join(out_chunks)
    pathlib.Path(out_txt).write_text(joined, encoding="utf-8")
    print(f"[ok] wrote {out_txt} ({len(joined):,} chars)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python corpus/combine_txts.py <dir_of_books> <out.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
