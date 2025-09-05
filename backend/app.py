# app.py — small FastAPI chat API with optional RAG (hybrid lexical+vector)
# Behavior:
# - Normal chat by default (no citations).
# - If the user clearly asks for a verse/scripture/quote, we run hybrid retrieval
#   and return ONE relevant passage with [CIT:<ref>], then brief guidance.
# - If there is no meaningful match, we skip RAG and just chat.

import os, re, json, uuid, math, random
from typing import Dict, List, Optional, Any, Tuple, Generator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- OpenAI (>=1.0) ----
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# ========= Session state (in-memory) =========
class SessionState:
    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = [{"role":"system","content":(
            "You are Fight Chaplain: calm, concise, and encouraging. "
            "Offer practical corner-coach guidance with spiritual grounding. "
            "Keep answers tight; avoid filler."
        )}]
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

# ========= Index loading =========
from pathlib import Path
def _candidate_index_dirs() -> List[Path]:
    here = Path(__file__).parent.resolve()
    env = os.getenv("RAG_DIR", "").strip()
    cands = []
    if env: cands.append(Path(env))
    # common layouts:
    cands += [
        here / "indexes",
        here / "backend" / "indexes",
        Path("/opt/render/project/src/indexes"),  # Render default checkout path
    ]
    # dedupe while preserving order
    uniq = []
    seen = set()
    for p in cands:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists(): return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                t = obj.get("text"); e = obj.get("embedding")
                if t is None or e is None: continue
                rows.append({
                    "id": obj.get("id") or obj.get("ref") or "",
                    "ref": obj.get("ref") or obj.get("book") or obj.get("id") or "",
                    "source": obj.get("source") or "",
                    "text": t,
                    "embedding": e,
                })
            except Exception:
                continue
    return rows

def find_and_load_corpora() -> Dict[str, List[Dict[str, Any]]]:
    files = ("bible.jsonl", "quran.jsonl", "talmud.jsonl")
    corpora: Dict[str, List[Dict[str, Any]]] = {"bible": [], "quran": [], "talmud": []}
    for base in _candidate_index_dirs():
        for name in files:
            p = base / name
            if p.exists() and p.stat().st_size > 0:
                key = name.split(".")[0]
                if not corpora[key]:  # first win
                    corpora[key] = load_jsonl(p)
    return corpora

CORPORA: Dict[str, List[Dict[str, Any]]] = find_and_load_corpora()

def corpus_counts() -> Dict[str,int]:
    return {k: len(v) for k,v in CORPORA.items()}

# ========= Tiny text/metric helpers =========
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
        x = a[i]; y = b[i]
        s += x*y; na += x*x; nb += y*y
    if na == 0 or nb == 0: return 0.0
    return s / math.sqrt(na*nb)

# ========= OpenAI wrappers =========
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
        # dev echo
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

# ========= RAG trigger (only when clearly asked) =========
ASK_WORDS = ["verse","scripture","psalm","quote","passage","ayah","surah","quran","bible","talmud","tractate"]
def wants_retrieval(msg: str) -> bool:
    m = f" {msg.lower()} "
    if any(f" {w} " in m for w in ASK_WORDS): return True
    if "give me a verse" in m or "share a verse" in m or "quote from" in m: return True
    return False

# ========= Hybrid search (lexical + vector) with “no signal → skip” =========
def hybrid_search(query: str, corpus_name: str, top_k: int = 6) -> List[Dict[str,Any]]:
    docs = CORPORA.get(corpus_name, [])
    if not docs: return []

    q_tokens = set(tokenize(query))

    # lexical
    lex_scores: List[Tuple[float,int]] = []
    any_lex = False
    for i, d in enumerate(docs):
        t = d.get("text","")
        if not t:
            lex_scores.append((0.0, i)); continue
        js = jaccard(q_tokens, set(tokenize(t)))
        if js > 0.0: any_lex = True
        lex_scores.append((js, i))

    # vector
    q_emb = embed_query(query)
    vec_scores: List[Tuple[float,int]] = []
    any_vec = False
    if q_emb:
        for i, d in enumerate(docs):
            e = d.get("embedding")
            s = cos(q_emb, e) if isinstance(e, list) else 0.0
            if s > 0.05: any_vec = True   # tiny positive signal
            vec_scores.append((s, i))
    else:
        vec_scores = [(0.0, i) for i in range(len(docs))]

    # No lexical match and no vector signal → skip RAG
    if not any_lex and not any_vec:
        return []

    # z-score normalize & blend
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
    out = [docs[i] for _, i in blended[:top_k]]
    return out

def attach_citation(text: str, hit: Optional[Dict[str,Any]]) -> str:
    if not hit: return text
    ref = (hit.get("ref") or "").strip()
    if not ref: return text
    return f'{text} [CIT: {ref}]'

# ========= SSE helper =========
def sse_event(event: Optional[str], data: str) -> str:
    if event:
        return f"event: {event}\n" + "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"
    return "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"

# ========= App + CORS =========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ========= Routes =========
@app.get("/")
def root():
    return PlainTextResponse("ok")

@app.get("/health")
def health():
    return PlainTextResponse("OK")

@app.get("/diag_rag")
def diag():
    return {"ok": True, "corpora": corpus_counts()}

def _sys_with_rollup(s: SessionState) -> Dict[str,str]:
    base = s.history[0]["content"]
    if s.rollup:
        base = base + "\nSESSION_SUMMARY: " + s.rollup
    return {"role":"system","content":base}

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

    # push user turn
    s.history.append({"role":"user","content":message})
    if len(s.history) > 40:
        s.history = s.history[:1] + s.history[-39:]

    use_rag = wants_retrieval(message)
    sources: List[Dict[str,Any]] = []

    if use_rag:
        # pick corpus by hinting from message (super simple)
        hint = message.lower()
        if "quran" in hint or "surah" in hint or "ayah" in hint:
            corp = "quran"
        elif "talmud" in hint or "tractate" in hint or "mishnah" in hint or "gemara" in hint:
            corp = "talmud"
        else:
            corp = "bible"
        hits = hybrid_search(message, corp, top_k=6)
        if hits:
            top = hits[0]
            verse = attach_citation(top.get("text",""), top)
            # return the verse + brief guidance via model
            ctx = (
                "RETRIEVED PASSAGES (quote at most one verbatim with [CIT:id], then add brief guidance):\n"
                f"- [{corp}] [CIT:{top.get('ref','')}] {top.get('text','').strip().replace('\n',' ')}"
            )
            sys_msg = _sys_with_rollup(s)
            sys_msg["content"] += "\n\n" + ctx
            messages = [sys_msg] + s.history[1:]
            out = call_openai(messages, stream=False)
            reply = out if isinstance(out, str) else "".join(list(out))
            # ensure at least the verse is present
            if verse not in reply:
                reply = verse + "\n\n" + reply
            s.history.append({"role":"assistant","content":reply})
            _maybe_rollup(s)
            return JSONResponse({"response": reply, "sid": sid,
                                 "sources":[{"ref": top.get("ref",""), "source": top.get("source","")}]})
        # fallthrough to normal chat if no match

    # normal chat
    out = call_openai([_sys_with_rollup(s)] + s.history[1:], stream=False)
    reply = out if isinstance(out, str) else "".join(list(out))
    s.history.append({"role":"assistant","content":reply})
    _maybe_rollup(s)
    return JSONResponse({"response": reply, "sid": sid, "sources": []})

@app.get("/chat_sse")
async def chat_sse(request: Request, q: str, sid: Optional[str] = None):
    base = Response()
    if sid: request.headers.__dict__.setdefault("_list", []).append((b"x-session-id", sid.encode()))
    sid2, s = get_or_create_sid(request, base)

    # push user turn
    s.history.append({"role":"user","content":q})
    if len(s.history) > 40:
        s.history = s.history[:1] + s.history[-39:]

    use_rag = wants_retrieval(q)

    def gen() -> Generator[str, None, None]:
        yield ": connected\n\n"
        yield sse_event("ping", "hi")

        if use_rag:
            hint = q.lower()
            if "quran" in hint or "surah" in hint or "ayah" in hint:
                corp = "quran"
            elif "talmud" in hint or "tractate" in hint or "mishnah" in hint or "gemara" in hint:
                corp = "talmud"
            else:
                corp = "bible"
            hits = hybrid_search(q, corp, top_k=6)
            if hits:
                top = hits[0]
                verse = attach_citation(top.get("text",""), top)
                ctx = (
                    "RETRIEVED PASSAGES (quote at most one verbatim with [CIT:id], then add brief guidance):\n"
                    f"- [{corp}] [CIT:{top.get('ref','')}] {top.get('text','').strip().replace('\n',' ')}"
                )
                sys_msg = _sys_with_rollup(s)
                sys_msg["content"] += "\n\n" + ctx
                stream = call_openai([sys_msg] + s.history[1:], stream=True)  # type: ignore
                buf = ""
                seen_any = False
                for piece in stream:
                    seen_any = True
                    buf += piece
                    if re.search(r"[.!?]\s+$", buf) or len(buf) >= 160:
                        yield f"data: {json.dumps({'text': buf.strip()}, ensure_ascii=False)}\n\n"
                        buf = ""
                if not seen_any:
                    yield f"data: {json.dumps({'text': verse}, ensure_ascii=False)}\n\n"
                elif buf:
                    yield f"data: {json.dumps({'text': buf.strip()}, ensure_ascii=False)}\n\n"
                s.history.append({"role":"assistant","content":""})
                _maybe_rollup(s)
                yield sse_event("done", json.dumps({"sid": sid2}))
                return
            # else fall through to normal chat

        # normal chat stream
        stream = call_openai([_sys_with_rollup(s)] + s.history[1:], stream=True)  # type: ignore
        buf = ""
        for piece in stream:
            buf += piece
            if re.search(r"[.!?]\s+$", buf) or len(buf) >= 160:
                yield f"data: {json.dumps({'text': buf.strip()}, ensure_ascii=False)}\n\n"
                buf = ""
        if buf:
            yield f"data: {json.dumps({'text': buf.strip()}, ensure_ascii=False)}\n\n"
        s.history.append({"role":"assistant","content":""})
        _maybe_rollup(s)
        yield sse_event("done", json.dumps({"sid": sid2}))

    headers = dict(base.headers)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# Keep GET stream “waker” for Render
@app.get("/chat_sse_get")
def chat_sse_get():
    def gen():
        yield ": connected\n\n"
        for t in ["This is a GET stream endpoint.\n"]:
            yield sse_event(None, t)
    return StreamingResponse(gen(), media_type="text/event-stream")
