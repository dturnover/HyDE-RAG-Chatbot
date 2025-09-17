import sys, re, json, urllib.parse, pathlib, requests

PREFERRED_SUFFIXES = [
    "_djvu.txt",   # OCR text alongside DjVu
    ".txt",        # sometimes a plain .txt exists
]

def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def guess_identifier(arg: str) -> str:
    # Accept a bare identifier OR any archive.org URL
    m = re.search(r"/(details|download|stream)/([^/?#]+)", arg)
    return m.group(2) if m else arg

def fetch_metadata(sess: requests.Session, ident: str) -> dict:
    url = f"https://archive.org/metadata/{ident}"
    r = sess.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def choose_text_file(meta: dict) -> str | None:
    files = meta.get("files") or []
    # collect candidates with preferred suffixes (case-insensitive)
    cands = []
    for f in files:
        name = f.get("name") or ""
        lname = name.lower()
        for suf in PREFERRED_SUFFIXES:
            if lname.endswith(suf):
                cands.append(name)
                break
    # prefer the longest (often the full book vs. part)
    if cands:
        return sorted(cands, key=len, reverse=True)[0]
    return None

def download_text(sess: requests.Session, ident: str, filename: str) -> str:
    # Use /download/<id>/<filename> with proper URL-encoding
    quoted = urllib.parse.quote(filename)
    url = f"https://archive.org/download/{ident}/{quoted}"
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def main(arg: str, out_path: str):
    ident = guess_identifier(arg)
    sess = requests.Session()
    # a reasonable UA helps avoid overzealous anti-bot heuristics
    sess.headers.update({"User-Agent": "ia-fetch-txt/1.0 (+https://archive.org)"})

    try:
        meta = fetch_metadata(sess, ident)
    except Exception as e:
        print(f"[fail] metadata fetch: {e}")
        sys.exit(2)

    fname = choose_text_file(meta)
    if not fname:
        print("[fail] no *_djvu.txt or .txt file advertised in metadata")
        # As a last resort, if the user pasted a full stream URL, try it as-is:
        if arg.startswith("http"):
            try:
                r = sess.get(arg, timeout=60)
                r.raise_for_status()
                body = r.text
                body = normalize(body)
                pathlib.Path(out_path).write_text(body, encoding="utf-8")
                print(f"[ok] wrote {out_path} (from provided URL; {len(body):,} chars)")
                return
            except Exception as e:
                print(f"[fallback failed] {e}")
        sys.exit(3)

    try:
        print(f"[info] downloading: {fname}")
        txt = download_text(sess, ident, fname)
        txt = normalize(txt)
        pathlib.Path(out_path).write_text(txt, encoding="utf-8")
        print(f"[ok] wrote {out_path} ({len(txt):,} chars)")
    except Exception as e:
        print(f"[fail] download text: {e}")
        sys.exit(4)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python tools/ia_fetch_txt.py <archive_id_or_url> <out.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
