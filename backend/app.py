import os, json, re, math, uuid
from typing import List, Dict, Any, Optional, Tuple, Generator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Basics ----------------

app = FastAPI()

# Open CORS so the GH Pages front-end works
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

INDEX_DIR = Path(__file__).parent / "indexes"
RAW_BASE  = "https://raw.githubusercontent.com/dturnover/LLM-API-DEMO/main/backend/indexes"

FILES = {
    "bible":  INDEX_DIR / "bible.jsonl",
    "quran":  INDEX_DIR / "quran.jsonl",
    "talmud": INDEX_DIR / "talmud.jsonl",
}

_token = re.compile(r"[A-Za-z0-9]+")

def tokenize(s: str) -> List[str]:
    return _token.findall(s.lower())

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists(): return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "text" in obj:
                    obj.setdefault("ref", obj.get("ref") or obj.get("id") or "")
                    obj.setdefault("source", obj.get("source") or "")
                    rows.append(obj)
            except Exception:
                continue
    return rows

CORPORA: Dict[str, List[Dict[str, Any]]] = {name: load_jsonl(p) for name, p in FILES.items()}

def counts() -> Dict[str,int]:
    return {k: len(v) for k, v in CORPORA.items()}

def sizes() -> Dict[str,int]:
    out: Dict[str,int] = {}
    for k, p in FILES.items():
        try: out[k] = p.stat().st_size if p.exists() else 0
        except Exception: out[k] = -1
    return out

# ---------------- Optional hydration (pull from GitHub raw) ----------------

def _fetch_to(path: Path, url: str) -> Tuple[bool, str]:
    import urllib.request, tempfile, shutil
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False) as tmpf:
            tmp = Path(tmpf.name)
        with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out)
        tmp.replace(path)
        return True, f"saved {path.name} ({path.stat().st_size} bytes)"
    except Exception as e:
        try:
            if 'tmp' in locals() and tmp.exists():
                tmp.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass
        return False, f"error: {e}"

def rehydrate_one(name: str) -> Tuple[bool, str]:
    if name not in FILES:
        return False, f"unknown corpus {name}"
    path = FILES[name]
    if path.exists() and path.stat().st_size > 4096:
        return True, f"exists ({path.stat().st_size} bytes)"
    url = f"{RAW_BASE}/{name}.jsonl"
    return _fetch_to(path, url)

def hot_reload():
    for k, p in FILES.items():
        CORPORA[k] = load_jsonl(p)

# ---------------- Search ----------------

def search_lexical(query: str, corpus: str, top_k: int = 1) -> List[Dict[str, Any]]:
    docs = CORPORA.get(corpus, [])
    if not docs: return []
    qtok = set(tokenize(query))
    scored = []
    for i, d in enumerate(docs):
        t = d.get("text", "")
        dtok = set(tokenize(t))
        scored.append((jaccard(qtok, dtok), i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [docs[i] for _, i in scored[:top_k]]

# ---------------- Routes ----------------

@app.get("/")
def root():
    return PlainTextResponse("ok")

@app.get("/health")
def health():
    return PlainTextResponse("OK")

@app.get("/diag_rag")
def diag_rag():
    return {"ok": True, "counts": counts(), "sizes": sizes()}

@app.post("/rehydrate")
def rehydrate(body: Optional[Dict[str, Any]] = None):
    names = (body or {}).get("names") if isinstance(body, dict) else None
    if not names: names = list(FILES.keys())
    results = {}
    for n in names:
        ok, msg = rehydrate_one(n)
        results[n] = msg
    hot_reload()
    return {"ok": True, "results": results, "counts": counts(), "sizes": sizes()}

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    message = (body.get("message") or "").strip()
    corpus  = (body.get("corpus") or "bible").strip().lower()
    if corpus not in FILES: corpus = "bible"

    hits = search_lexical(message, corpus, top_k=1)
    if hits:
        h = hits[0]
        txt = h.get("text", "")
        ref = h.get("ref", "")
        if ref: txt = f"{txt} [CIT: {ref}]"
        return JSONResponse({"response": txt, "sid": uuid.uuid4().hex, "sources": [{"ref": ref, "source": h.get("source","")}]})
    return JSONResponse({"response": "I couldn’t find a verse right now. (Index may be empty.)", "sid": uuid.uuid4().hex, "sources": []})

@app.get("/chat_sse")
def chat_sse(q: str, corpus: Optional[str] = "bible"):
    corpus = (corpus or "bible").lower()
    if corpus not in FILES: corpus = "bible"

    def gen() -> Generator[str, None, None]:
        yield ": connected\n\n"
        hits = search_lexical(q, corpus, top_k=1)
        if hits:
            h = hits[0]
            txt = h.get("text","")
            ref = h.get("ref","")
            if ref: txt = f"{txt} [CIT: {ref}]"
            # stream in sentences-ish
            buf = ""
            for w in txt.split():
                buf += w + " "
                if len(buf) >= 120 or re.search(r"[.!?]$", w):
                    yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
                    buf = ""
            if buf:
                yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
        else:
            yield f"data: {json.dumps({'text': 'No index available yet.'})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
