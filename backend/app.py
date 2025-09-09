# backend/app.py
# LLM API Demo — memory + rollup + hybrid RAG + robust SSE
# - Normal chat by default (no citations).
# - RAG triggers on explicit asks (verse/scripture/quote...) OR emotional context.
# - Faith is remembered in-session and routes retrieval to the matching corpus.
# - Only ONE verbatim passage is quoted with [CIT: ref] before guidance.
# - SSE streams immediately with heartbeats and proxy headers (fine-grained flush).

import os, re, json, uuid, math, asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- OpenAI client ----------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def oa_client() -> Optional[OpenAI]:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_query(text: str) -> Optional[List[float]]:
    cli = oa_client()
    if not cli:
        return None
    try:
        r = cli.embeddings.create(model=EMBED_MODEL, input=text)
        return r.data[0].embedding  # type: ignore
    except Exception:
        return None

def call_openai(messages: List[Dict[str,str]], stream: bool):
    cli = oa_client()
    if not cli:
        if stream:
            def _gen():
                for w in "Dev mode: OpenAI disabled on server.".split():
                    yield w + " "
            return _gen()
        return "⚠️ OpenAI disabled on server; running in dev mode."
    if stream:
        resp = cli.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
            stream=True
        )
        def _gen():
            for ev in resp:
                piece = ev.choices[0].delta.content or ""
                if piece:
                    yield piece
        return _gen()
    resp = cli.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.4)
    return resp.choices[0].message.content or ""

# ---------- Session ----------
class SessionState:
    def __init__(self) -> None:
        self.history: List[Dict[str,str]] = [{
            "role":"system",
            "content":(
                "You are Fight Chaplain: calm, concise, encouraging. "
                "Offer practical corner-coach guidance with spiritual grounding. "
                "Keep answers tight; avoid filler. "
                "When RETRIEVED PASSAGES are provided, quote AT MOST ONE verbatim "
                "with the provided [CIT: ref] tag, then give brief guidance."
            )
        }]
        self.rollup: Optional[str] = None
        self.faith: Optional[str] = None  # 'bible' | 'quran' | 'talmud'

SESSIONS: Dict[str, SessionState] = {}

def get_or_create_sid(request: Request, response: Response) -> Tuple[str, SessionState]:
    sid = request.headers.get("X-Session-Id") or request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie("sid", sid, httponly=True, samesite="none", secure=True, path="/")
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    return sid, SESSIONS[sid]

# ---------- Index loading ----------
def _candidate_index_dirs() -> List[Path]:
    here = Path(__file__).parent.resolve()
    env = os.getenv("RAG_DIR", "").strip()
    cands: List[Path] = []
    if env: cands.append(Path(env))
    cands += [
        here / "indexes",
        here / "backend" / "indexes",
        Path("/opt/render/project/src/indexes"),
    ]
    seen, out = set(), []
    for p in cands:
        if str(p) not in seen:
            out.append(p); seen.add(str(p))
    return out

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists(): return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            t, e = o.get("text"), o.get("embedding")
            if t is None or e is None:
                continue
            rows.append({
                "id": o.get("id") or o.get("ref") or "",
                "ref": o.get("ref") or o.get("book") or o.get("id") or "",
                "source": o.get("source") or "",
                "text": t,
                "embedding": e,
            })
    return rows

def find_and_load_corpora() -> Dict[str, List[Dict[str, Any]]]:
    files = ("bible.jsonl", "quran.jsonl", "talmud.jsonl")
    corpora: Dict[str, List[Dict[str, Any]]] = {"bible": [], "quran": [], "talmud": []}
    for base in _candidate_index_dirs():
        for name in files:
            p = base / name
            if p.exists() and p.stat().st_size > 0:
                key = name.split(".")[0]
                if not corpora[key]:
                    corpora[key] = load_jsonl(p)
    return corpora

CORPORA: Dict[str, List[Dict[str, Any]]] = find_and_load_corpora()
def corpus_counts() -> Dict[str,int]: return {k: len(v) for k,v in CORPORA.items()}

# ---------- Text utils ----------
_token = re.compile(r"[A-Za-z0-9]+")

def tokenize(s: str) -> List[str]:
    return _token.findall(s.lower())

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

def cos(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    s = 0.0; na = 0.0; nb = 0.0
    n = min(len(a), len(b))
    for i in range(n):
        x, y = a[i], b[i]
        s += x*y; na += x*x; nb += y*y
    if na == 0 or nb == 0: return 0.0
    return s / math.sqrt(na*nb)

# ---------- Faith memory ----------
FAITH_KEYWORDS = {
    "jewish": "talmud", "jew": "talmud", "hebrew": "talmud",
    "muslim": "quran", "islam": "quran",
    "christian": "bible", "catholic": "bible", "protestant": "bible",
}

def try_set_faith(msg: str, s: SessionState) -> None:
    m = msg.lower()
    blob = f" {m} "
    for k, corp in FAITH_KEYWORDS.items():
        if f" {k} " in blob or m.startswith(k) or m.endswith(k):
            s.faith = corp
            return

def detect_corpus(msg: str, s: SessionState) -> str:
    if s.faith: return s.faith
    m = msg.lower()
    if any(w in m for w in ["quran","surah","ayah"]): return "quran"
    if any(w in m for w in ["talmud","tractate","mishnah","gemara"]): return "talmud"
    return "bible"

# ---------- RAG triggers ----------
ASK_WORDS = ["verse","scripture","psalm","quote","passage","ayah","surah","quran","bible","talmud","tractate"]
EMO_WORDS = [
    "scared","afraid","anxious","nervous","panic","hurt","down","lost","depressed",
    "angry","worried","fight","injury","pain","tired","exhausted","grief","grieving",
    "lonely","alone","breakup","fear","doubt","stress","stressed"
]

def wants_retrieval(msg: str) -> bool:
    m = f" {msg.lower()} "
    if any(f" {w} " in m for w in ASK_WORDS): 
        return True
    if any(f" {w} " in m for w in EMO_WORDS):
        return True
    return False

# ---------- Hybrid search ----------
def hybrid_search(query: str, corpus_name: str, top_k: int = 6) -> List[Dict[str,Any]]:
    docs = CORPORA.get(corpus_name, [])
    if not docs: return []

    q_tokens = set(tokenize(query))

    lex_scores: List[Tuple[float,int]] = []
    any_lex = False
    for i, d in enumerate(docs):
        t = d.get("text","")
        js = jaccard(q_tokens, set(tokenize(t))) if t else 0.0
        if js > 0.0: any_lex = True
        lex_scores.append((js, i))

    q_emb = embed_query(query)
    any_vec = False
    if q_emb:
        vec_scores = []
        for i, d in enumerate(docs):
            e = d.get("embedding")
            s = cos(q_emb, e) if isinstance(e, list) else 0.0
            if s > 0.05: any_vec = True
            vec_scores.append((s, i))
    else:
        vec_scores = [(0.0, i) for i in range(len(docs))]

    if not any_lex and not any_vec:
        return []  # no signal → skip RAG altogether

    def zscore(lst: List[Tuple[float,int]]) -> Dict[int,float]:
        vals = [x for x,_ in lst]
        if not vals: return {}
        mu = sum(vals)/len(vals)
        sd = math.sqrt(sum((x-mu)**2 for x in vals)/len(vals)) or 1.0
        return {j:(x-mu)/sd for x,j in lst}

    L = zscore(lex_scores); V = zscore(vec_scores)
    alpha = 0.6
    blend = [(alpha*L.get(i,0.0) + (1-alpha)*V.get(i,0.0), i) for i in range(len(docs))]
    blend.sort(reverse=True)
    return [docs[i] for _, i in blend[:top_k]]

def attach_citation(text: str, hit: Optional[Dict[str,Any]]) -> str:
    """
    Render a single, compact citation like [Quran p25] / [Bible p1245] / [Talmud p1531].
    (We only have the short ref id; if you later add full book:verse, format it here.)
    """
    if not hit: 
        return text
    ref = (hit.get("ref") or "").strip()
    src = (hit.get("source") or "").strip().title() or "Source"
    if not ref:
        return text
    return f"{text} [{src} {ref}]"

# ---------- SSE helpers ----------
def sse_event(event: Optional[str], data: str) -> str:
    if event:
        return f"event: {event}\n" + "\n".join(f"data: {ln}" for ln in data.splitlines()) + "\n\n"
    return "\n".join(f"data: {ln}" for ln in data.splitlines()) + "\n\n"

def sse_headers_for_origin(origin: Optional[str]) -> Dict[str,str]:
    h = {"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"}
    if origin and (origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS):
        h["Access-Control-Allow-Origin"] = origin if origin in ALLOWED_ORIGINS else "*"
        h["Access-Control-Allow-Credentials"] = "true"
        h["Access-Control-Expose-Headers"] = "*"
    return h

# ---------- FastAPI + CORS ----------
app = FastAPI()

ALLOWED_ORIGINS = {
    "https://dturnover.github.io",
    "http://localhost:3000","http://127.0.0.1:3000",
    "http://localhost:5500","http://127.0.0.1:5500",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

@app.middleware("http")
async def cors_exact_origin(request: Request, call_next):
    origin = request.headers.get("origin", "")
    resp = await call_next(request)
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        req_hdrs = request.headers.get("access-control-request-headers")
        resp.headers["Access-Control-Allow-Headers"] = req_hdrs or "*"
        resp.headers["Access-Control-Expose-Headers"] = "*"
    return resp

@app.options("/{rest_of_path:path}")
def options_preflight(request: Request, rest_of_path: str):
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in ALLOWED_ORIGINS:
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Vary": "Origin",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers") or "*",
            "Access-Control-Max-Age": "600",
        }
    return PlainTextResponse("OK", headers=headers)

# ---------- Small helpers ----------
def _sys_with_rollup(s: SessionState) -> Dict[str,str]:
    base = s.history[0]["content"]
    if s.rollup:
        base += "\nSESSION_SUMMARY: " + s.rollup
    return {"role":"system","content":base}

def _maybe_rollup(s: SessionState) -> None:
    if len(s.history) >= 24 and not s.rollup:
        user_lines = [m["content"] for m in s.history if m["role"] == "user"]
        if user_lines:
            s.rollup = ("Themes: " + "; ".join(user_lines[-6:]))[:500]

# ---------- Routes ----------
@app.get("/")
def root(): return PlainTextResponse("ok")

@app.get("/health")
def health(): return PlainTextResponse("OK")

@app.get("/diag_rag")
def diag(): return {"ok": True, "corpora": corpus_counts()}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    msg = (body.get("message") or "").strip()

    base = Response()
    sid, s = get_or_create_sid(request, base)

    try_set_faith(msg, s)  # update faith memory
    s.history.append({"role":"user","content":msg})
    if len(s.history) > 40:
        s.history = s.history[:1] + s.history[-39:]

    if wants_retrieval(msg):
        corp = detect_corpus(msg, s)
        hits = hybrid_search(msg, corp, top_k=6)
        if hits:
            top = hits[0]
            verse = attach_citation(top.get("text",""), top)
            ctx = (
                "RETRIEVED PASSAGES (quote AT MOST ONE verbatim with [CIT: ref], "
                "then add brief guidance):\n"
                f"- [{corp}] [CIT: {top.get('ref','')}] {top.get('text','').strip().replace('\n',' ')}"
            )
            sys_msg = _sys_with_rollup(s)
            sys_msg["content"] += "\n\n" + ctx
            messages = [sys_msg] + s.history[1:]
            out = call_openai(messages, stream=False)
            reply = out if isinstance(out, str) else "".join(list(out))
            if verse not in reply:
                reply = verse + "\n\n" + reply
            s.history.append({"role":"assistant","content":reply})
            _maybe_rollup(s)
            return JSONResponse(
                {"response": reply, "sid": sid,
                 "sources":[{"ref": top.get("ref",""), "source": top.get("source","")}]},
                headers={"X-Session-Id": sid}
            )

    # Normal chat path
    out = call_openai([_sys_with_rollup(s)] + s.history[1:], stream=False)
    reply = out if isinstance(out, str) else "".join(list(out))
    s.history.append({"role":"assistant","content":reply})
    _maybe_rollup(s)
    return JSONResponse({"response": reply, "sid": sid, "sources": []}, headers={"X-Session-Id": sid})

@app.get("/chat_sse")
async def chat_sse(request: Request, q: str, sid: Optional[str] = None):
    base = Response()
    try:
        if sid and sid in SESSIONS:
            sid2, s = sid, SESSIONS[sid]
        else:
            sid2, s = get_or_create_sid(request, base)
    except Exception:
        sid2, s = uuid.uuid4().hex, SessionState()
        SESSIONS[sid2] = s

    # update faith memory now, before any retrieval
    try_set_faith(q, s)
    origin = request.headers.get("origin")

    async def agen():
        # prelude so the socket isn't 200/empty
        yield b": connected\n\n"
        yield b"event: ping\ndata: hi\n\n"

        last_hb = asyncio.get_event_loop().time()

        def chunk_emit(txt: str) -> bytes:
            # One SSE data frame with JSON {text: "..."}
            return ("data: " + json.dumps({"text": txt}) + "\n\n").encode("utf-8")

        try:
            # -------- RAG branch first if trigger fires (stream small chunks) --------
            if wants_retrieval(q):
                corp = detect_corpus(q, s)
                hits = hybrid_search(q, corp, top_k=6)
                if hits:
                    top = hits[0]
                    text = attach_citation(top.get("text",""), top)

                    pending = ""
                    last_flush = asyncio.get_event_loop().time()
                    for tok in text.split():
                        pending += tok + " "
                        now = asyncio.get_event_loop().time()
                        # FLUSH FAST: ~every 10-14 chars or 50 ms
                        if len(pending) >= 12 or (now - last_flush) >= 0.05:
                            yield chunk_emit(pending)
                            pending = ""
                            last_flush = now
                        if (now - last_hb) > 10:
                            yield b": hb\n\n"
                            last_hb = now
                        await asyncio.sleep(0)  # cooperative yield
                    if pending:
                        yield chunk_emit(pending)

                    # persist turn and end
                    s.history.append({"role":"user","content":q})
                    s.history.append({"role":"assistant","content":text})
                    _maybe_rollup(s)
                    yield ("event: done\ndata: " + json.dumps({"sid": sid2}) + "\n\n").encode("utf-8")
                    return

            # -------- Model streaming path (flush per piece) --------
            stream = call_openai(
                [_sys_with_rollup(s)] + s.history[1:] + [{"role":"user","content":q}],
                stream=True
            )

            assistant_out = ""
            pending = ""
            last_flush = asyncio.get_event_loop().time()

            for piece in stream:  # type: ignore
                if not piece:
                    continue
                pending += piece
                now = asyncio.get_event_loop().time()

                # FLUSH FAST: small buffer, newline, or every ~40 ms
                if len(pending) >= 8 or "\n" in pending or (now - last_flush) >= 0.04:
                    yield chunk_emit(pending)
                    assistant_out += pending
                    pending = ""
                    last_flush = now

                if (now - last_hb) > 10:
                    yield b": hb\n\n"
                    last_hb = now

                await asyncio.sleep(0)  # cooperative yield

            if pending:
                yield chunk_emit(pending)
                assistant_out += pending

            s.history.append({"role":"user","content":q})
            s.history.append({"role":"assistant","content":assistant_out.strip()})
            _maybe_rollup(s)
            yield ("event: done\ndata: " + json.dumps({"sid": sid2}) + "\n\n").encode("utf-8")

        except Exception as e:
            # Don’t drop the socket empty—tell the client why
            err = {"error": str(e)}
            yield ("event: error\ndata: " + json.dumps(err) + "\n\n").encode("utf-8")
            yield b": end\n\n"

    # headers: merge cookie + SSE/CORS + proxy hints
    hdrs = {**dict(base.headers), **sse_headers_for_origin(origin)}
    hdrs["Cache-Control"] = "no-cache"
    hdrs["X-Accel-Buffering"] = "no"

    return StreamingResponse(agen(), media_type="text/event-stream; charset=utf-8", headers=hdrs)

@app.get("/sse_test")
async def sse_test():
    async def agen():
        yield b": connected\n\n"
        for i in range(1, 6):
            await asyncio.sleep(0.4)
            yield f"data: tick {i}\n\n".encode("utf-8")
        yield b"event: done\ndata: ok\n\n"
    return StreamingResponse(agen(), media_type="text/event-stream; charset=utf-8")

# Minimal GET streamer to keep platforms happy
@app.get("/chat_sse_get")
def chat_sse_get():
    def gen():
        yield ": connected\n\n"
        yield sse_event(None, "This is a GET stream endpoint.")
    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")
