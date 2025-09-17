# tools/ia_fetch_txt.py
import sys, re, requests, pathlib

CANDIDATES = [
    # Most common: OCR text produced alongside DjVu
    "https://archive.org/download/{id}/{id}_djvu.txt",
    # Sometimes plain .txt exists
    "https://archive.org/download/{id}/{id}.txt",
    # Stream form also works on many items
    "https://archive.org/stream/{id}/{id}_djvu.txt",
]

def looks_like_text(s: str) -> bool:
    if not s or len(s) < 5000:              # too tiny = likely metadata
        return False
    if re.search(r"<(html|head|body|div|span)[^>]*>", s, re.I):  # html?
        return False
    return True

def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def main(identifier: str, out_path: str):
    sess = requests.Session()
    for url in [u.format(id=identifier) for u in CANDIDATES]:
        try:
            print(f"[try] {url}")
            r = sess.get(url, timeout=30)
            if r.status_code != 200:
                print(f"  -> {r.status_code}")
                continue
            txt = r.text
            if looks_like_text(txt):
                txt = normalize(txt)
                pathlib.Path(out_path).write_text(txt, encoding="utf-8")
                print(f"[ok] wrote {out_path} ({len(txt):,} chars)")
                return
            else:
                print("  -> content didn’t look like plain OCR text")
        except Exception as e:
            print(f"  -> error: {e}")
    print("[fail] none of the candidate URLs returned usable text")
    sys.exit(2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python tools/ia_fetch_txt.py <archive_id_or_stream_url> <out.txt>")
        sys.exit(1)
    ident = sys.argv[1]
    # Accept either a raw identifier or a full /stream/ URL
    m = re.search(r"/(stream|download)/([^/]+)/?", ident)
    if m:
        ident = m.group(2)
    main(ident, sys.argv[2])
