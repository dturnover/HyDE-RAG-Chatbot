# tools/html_folder_to_txt.py
import sys, pathlib, re
from bs4 import BeautifulSoup

def clean_text(s: str) -> str:
    # normalize whitespace, drop very short/boilerplate lines
    s = s.replace("\r\n","\n").replace("\r","\n")
    lines, keep = [], []
    for ln in s.split("\n"):
        t = ln.strip()
        if not t: 
            continue
        # skip navigation-y fluff
        if len(t) < 3: 
            continue
        if re.search(r'(copyright|all rights reserved|home|contact|site map)', t, re.I):
            continue
        keep.append(t)
    return "\n".join(keep) + "\n"

def main(src_dir, out_txt):
    p = pathlib.Path(src_dir)
    htmls = sorted([str(x) for x in p.glob("*.htm*")])
    out_parts = []
    print(f"[parse] {len(htmls)} html files")
    for fp in htmls:
        print("  +", pathlib.Path(fp).name)
        html = open(fp, "r", encoding="utf-8", errors="ignore").read()
        soup = BeautifulSoup(html, "html.parser")
        # prefer main/article/content-ish nodes; else body text
        node = soup.find(["main","article"]) or soup.find(id=re.compile("content|main", re.I)) or soup.body
        text = node.get_text("\n") if node else soup.get_text("\n")
        out_parts.append(clean_text(text))
    joined = "\n\n".join(out_parts)
    open(out_txt, "w", encoding="utf-8").write(joined)
    print(f"[ok] wrote {out_txt} ({len(joined):,} chars)")

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("usage: python tools/html_folder_to_txt.py <src_html_dir> <out.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
