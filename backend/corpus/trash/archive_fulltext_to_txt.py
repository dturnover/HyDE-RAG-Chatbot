# tools/archive_fulltext_to_txt.py
import sys, requests
from bs4 import BeautifulSoup

def main(url, out_txt):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pre = soup.find("pre")
    text = pre.get_text("\n") if pre else soup.get_text("\n")
    # normalize Windows/mac/newlines and trim
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    open(out_txt, "w", encoding="utf-8").write(text)
    print(f"[ok] wrote {out_txt} ({len(text):,} chars)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python tools/archive_fulltext_to_txt.py <archive_url> <out.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
