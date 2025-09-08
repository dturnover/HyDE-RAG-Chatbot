# app.py — FastAPI chat API with optional hybrid RAG + robust SSE
# Behavior:
# - Normal chat by default (no citations).
# - RAG triggers when the user explicitly asks for scripture/verse/quote
#   OR when an emotional cue is detected (quick path).
# - Retrieval uses hybrid lexical + vector scoring. If there is no signal,
#   we skip RAG and just chat.
# - Streaming (SSE) yields bytes immediately to avoid CL:0 / proxy buffering.

import os, re, json, uuid, math
from typing import Dict, List, Optional, Any, Tuple, Generator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- OpenAI (>=1.0) ----------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ---------- Session state ----------
class SessionState:
    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = [{
            "role":"system","content":(
                "You are Fight Chaplain: calm, concise, encouraging. "
                "Offer practical corner-coach guidance with spiritual grounding. "
                "Keep answers tight; avoid filler."
            )
        }]
        self.rollup: Optional[str] = None

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
from pathlib import Path

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
    seen, uniq = set(), []
    for p in cands:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                text = obj.get("text"); emb = obj.get("embedding")
                if text is None or emb is None: continue
                out.append({
                    "id": obj.get("id") or obj.get("ref") or "",
                    "ref": obj.get("ref") or obj.get("book") or obj.get("id") or "",
                    "source": obj.get("source") or "",
                    "text": text,
                    "embedding": emb,
                })
            except Exception:
                continue
    return out

def find_and_load_corpora() -> Dict[str, List[Dict[str, Any]]]:
    names = ("bible.jsonl", "quran.jsonl", "talmud.jsonl")
    corpora: Dict[str, List[Dict[str, Any]]] = {"bible": [], "quran": [], "talmud": []}
    for base in _candidate_index_dirs():
        for nm in names:
            p = base / nm
            if p.exists() and p.stat().st_size > 0:
                key = nm.split(".")[0]
                if not corpora[key]:
                    corpora[key] = load_jsonl(p)
    return corpora

CORPORA: Dict[str, List[Dict[str, Any]]] = find_and_load_corpora()

def corpus_counts() -> Dict[str,int]:
    return {k: len(v) for k,v in CORPORA.items()}

# ---------- Tiny text/metric helpers ----------
_token = re.compile(r"[A-Za-z0-9]+")

def tokenize(s: str) -> List[str]:
    return _token.findall(s.lower())

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def cos(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    s = 0.0; na = 0.0; nb = 0.0
    n = min(len(a), len(b))
    for i in range(n):
        x, y = a[i], b[i]
        s += x*y; na += x*x; nb += y*y
    if na == 0 or nb == 0: return 0.0
    return s / math.sqrt(na*nb)

# ---------- OpenAI wrappers ----------
def oa_client() -> Optional[OpenAI]:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_query(text: str) -> Optional[List[float]]:
    cli = oa_client()
    if not cli: return None
    try:
        resp = cli.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding  # type: ignore
    except Exception:
        return None

def call_openai(messages: List[Dict[str,str]], stream: bool):
    cli = oa_client()
    if not cli:
        if stream:
            def _g():
                for w in "Dev mode: OpenAI disabled on server.".split():
                    yield w + " "
            return _g()
        else:
            return "⚠️ OpenAI disabled on server; running in dev mode."
    if stream:
        resp = cli.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.4, stream=True)
        def _gen():
            for ev in resp:
                piece = ev.choices[0].delta.content or ""
                if piece: yield piece
        return _gen()
    else:
        resp = cli.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.4)
        return resp.choices[0].message.content or ""

# ---------- Retrieval trigger + corpus detection ----------
ASK_WORDS = ["verse","scripture","psalm","quote","passage","ayah","surah","quran","bible","talmud","tractate"]
EMO_WORDS = ["scared","afraid","anxious","nervous","panic","hurt","down","lost","depressed","angry","worried","fight","injury","pain","tired","exhausted","grateful"]

def wants_retrieval(msg: str) -> bool:
    m = f" {msg.lower()} "
    if any(f" {w} " in m for w in ASK_WORDS): return True
    if any(f" {w} " in m for w in EMO_WORDS): return True
    if "give me a verse" in m or "share a verse" in m or "quote from" in m: return True
    return False

def detect_corpus(msg: str) -> str:
    m = msg.lower()
    if any(w in m for w in ["quran","surah","ayah"]): return "quran"
    if any(w in m for w in ["talmud","tractate","mishnah","gemara"]): return "talmud"
    return "bible"

# ---------- Hybrid search ----------
def hybrid_search(query: str, corpus_name: str, top_k: int = 6) -> List[Dict[str,Any]]:
    docs = CORPORA.get(corpus_name, [])
    if not docs: return []

    q_tokens = set(tokenize(query))

    # lexical
    any_lex = False
    lex_scores: List[Tuple[float,int]] = []
    for i, d in enumerate(docs):
        t = d.get("text","")
        js = jaccard(q_tokens, set(tokenize(t))) if t else 0.0
        if js > 0.0: any_lex = True
        lex_scores.append((js, i))

    # vector
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
        return []

    def zscore(lst: List[Tuple[float,int]]) -> Dict[int,float]:
        vals = [x for x,_ in lst]
        if not vals: return {}
        mu = sum(vals)/len(vals)
        sd = math.sqrt(sum((x-mu)**2 for x in vals)/len(vals)) or 1.0
        return {j:(x-mu)/sd for x,j in lst}

    L = zscore(lex_scores); V = zscore(vec_scores)
    alpha = 0.6
    blended = [(alpha*L.get(i,0.0) + (1-alpha)*V.get(i,0.0), i) for i in range(len(docs))]
    blended.sort(reverse=True)
    return [docs[i] for _, i in blended[:top_k]]

# ---------- Citation helpers ----------
CIT_RE = re.compile(r"\[CIT:\s*[^]]+\]")

def attach_citation(text: str, hit: Optional[Dict[str,Any]]) -> str:
    if not hit: return text
    ref = (hit.get("ref") or "").strip()
    if not ref: return text
    return f"{text} [CIT: {ref}]"

def keep_one_citation(text: str) -> str:
    """Keep the first [CIT: …] and remove any later ones; normalize spacing."""
    m = CIT_RE.search(text)
    if not m:
        return text
    head = text[:m.end()]
    tail = CIT_RE.sub("", text[m.end():])
    return re.sub(r"\s{2,}", " ", head + tail).strip()

# ---------- SSE helpers ----------
def sse_headers_for_origin(origin: Optional[str]) -> Dict[str,str]:
    h = {
        "Cache-Control":"no-cache",
        "Connection":"keep-alive",
        "X-Accel-Buffering":"no",
    }
    if origin and (origin in ALLOWED_ORIGINS):
        h["Access-Control-Allow-Origin"] = origin
        h["Access-Control-Allow-Credentials"] = "true"
        h["Access-Control-Expose-Headers"] = "*"
    return h

# ---------- FastAPI + CORS ----------
app = FastAPI()

ALLOWED_ORIGINS = {
    "https://dturnover.github.io",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5500", "http://127.0.0.1:5500",
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
    origin = request.headers.get("origin","")
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

@app.options("/{rest:path}")
def options_preflight(request: Request, rest: str):
    origin = request.headers.get("origin","")
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

# ---------- Routes ----------
@app.get("/")
def root(): return PlainTextResponse("ok")

@app.get("/health")
def health(): return PlainTextResponse("OK")

@app.get("/diag_rag")
def diag(): return {"ok": True, "corpora": corpus_counts()}

def _sys_with_rollup(s: SessionState) -> Dict[str,str]:
    base = s.history[0]["content"]
    if s.rollup:
        base += "\nSESSION_SUMMARY: " + s.rollup
    return {"role":"system","content": base}

def _maybe_rollup(s: SessionState) -> None:
    if len(s.history) >= 24 and not s.rollup:
        user_lines = [m["content"] for m in s.history if m["role"]=="user"]
        if user_lines:
            s.rollup = "Themes: " + "; ".join(user_lines[-6:])[:500]

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()

    base = Response()
    sid, s = get_or_create_sid(request, base)

    # Add user turn
    s.history.append({"role":"user","content":message})
    if len(s.history) > 40:
        s.history = s.history[:1] + s.history[-39:]

    sources: List[Dict[str,Any]] = []
    if wants_retrieval(message):
        corp = detect_corpus(message)
        hits = hybrid_search(message, corp, top_k=6)
        if hits:
            top = hits[0]
            verse = attach_citation(top.get("text",""), top)
            ctx = (
                "RETRIEVED PASSAGES (quote at most one verbatim with [CIT:id], then give brief guidance):\n"
                f"- [{corp}] [CIT:{top.get('ref','')}] {top.get('text','').strip().replace('\n',' ')}"
            )
            sys_msg = _sys_with_rollup(s)
            sys_msg["content"] += "\n\n" + ctx
            messages = [sys_msg] + s.history[1:]
            out = call_openai(messages, stream=False)
            reply = out if isinstance(out, str) else "".join(list(out))
            if verse not in reply:
                reply = verse + "\n\n" + reply
            reply = keep_one_citation(reply)
            s.history.append({"role":"assistant","content":reply})
            _maybe_rollup(s)
            return JSONResponse({"response": reply, "sid": sid,
                                 "sources":[{"ref": top.get("ref",""), "source": top.get("source","")}]})

    # Normal chat
    out = call_openai([_sys_with_rollup(s)] + s.history[1:], stream=False)
    reply = out if isinstance(out, str) else "".join(list(out))
    s.history.append({"role":"assistant","content":reply})
    _maybe_rollup(s)
    return JSONResponse({"response": reply, "sid": sid, "sources": []})

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

    origin = request.headers.get("origin")

    def gen() -> Generator[str, None, None]:
        # Yield immediately so proxies treat as streaming (no CL:0)
        yield ": connected\n\n"
        yield "event: ping\ndata: hi\n\n"

        try:
            m = q.lower()

            # RAG quick path: explicit ask or emotional language
            if wants_retrieval(q):
                corp = detect_corpus(q)
                hits = hybrid_search(q, corp, top_k=6)
                if hits:
                    top = hits[0]
                    text = attach_citation(top.get("text",""), top)
                    # chunk stream
                    buf = ""
                    for token in text.split():
                        buf += token + " "
                        if re.search(r"[.!?]\s+$", buf) or len(buf) >= 200:
                            yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
                            buf = ""
                    if buf:
                        yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
                    s.history.append({"role":"user","content":q})
                    s.history.append({"role":"assistant","content":text})
                    _maybe_rollup(s)
                    yield f"event: done\ndata: {json.dumps({'sid': sid2})}\n\n"
                    return

            # Normal LLM streaming
            stream = call_openai(
                [_sys_with_rollup(s)] + s.history[1:] + [{"role":"user","content":q}],
                stream=True
            )  # type: ignore

            buf = ""
            for piece in stream:
                buf += piece
                if re.search(r"[.!?]\s+$", buf) or len(buf) >= 200:
                    yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
                    buf = ""
            if buf:
                yield f"data: {json.dumps({'text': buf.strip()})}\n\n"

            s.history.append({"role":"user","content":q})
            s.history.append({"role":"assistant","content":""})
            _maybe_rollup(s)
            yield f"event: done\ndata: {json.dumps({'sid': sid2})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    headers = {**dict(base.headers), **sse_headers_for_origin(origin)}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# Keep simple GET stream (warmup / diagnostics)
@app.get("/chat_sse_get")
def chat_sse_get():
    def g():
        yield ": connected\n\n"
        yield "data: This is a GET stream endpoint.\n\n"
    return StreamingResponse(g(), media_type="text/event-stream")
