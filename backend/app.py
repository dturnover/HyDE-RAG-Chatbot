import os, re, json, uuid, math, random
from typing import Dict, List, Optional, Any, Tuple, Generator
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# -------- OpenAI (>=1.0) ----------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# -------- Session state ----------
class SessionState:
    def __init__(self) -> None:
        self.faith: Optional[str] = None
        self.facts: Dict[str, str] = {}
        self.history: List[Dict[str, str]] = []
        self.seen_refs: set[str] = set()
        self.rollup: Optional[str] = None

SESSIONS: Dict[str, SessionState] = {}


def embed_query(text: str) -> Optional[List[float]]:
    cli = oa_client()
    if not cli: return None
    try:
        resp = cli.embeddings.create(model=EMBED_MODEL, input=text)  # <-- use env
        return resp.data[0].embedding  # type: ignore
    except Exception:
        return None

def get_or_create_sid(request: Request, response: Response) -> Tuple[str, SessionState]:
    sid = request.headers.get("X-Session-Id") or request.cookies.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie("sid", sid, httponly=True, samesite="none", secure=True, path="/")
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    return sid, SESSIONS[sid]

# -------- Index loading ----------
# NOTE: this points to backend/indexes (app.py lives in backend/app/)
INDEX_DIR = Path(__file__).resolve().parent.parent / "indexes"

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

CORPORA_FILES = {
    "bible":  INDEX_DIR / "bible.jsonl",
    "quran":  INDEX_DIR / "quran.jsonl",
    "talmud": INDEX_DIR / "talmud.jsonl",
}
CORPORA: Dict[str, List[Dict[str, Any]]] = {k: load_jsonl(v) for k, v in CORPORA_FILES.items()}

def corpus_counts() -> Dict[str,int]:
    return {k: len(v) for k,v in CORPORA.items()}

# -------- Text helpers ----------
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
    for x,y in zip(a,b):
        s += x*y; na += x*x; nb += y*y
    if na == 0 or nb == 0: return 0.0
    return s / (na**0.5 * nb**0.5)

# -------- OpenAI wrappers ----------
def oa_client() -> Optional[OpenAI]:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_query(text: str) -> Optional[List[float]]:
    cli = oa_client()
    if not cli: return None
    try:
        resp = cli.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding  # type: ignore
    except Exception:
        return None

def call_openai(messages: List[Dict[str,str]], stream: bool):
    cli = oa_client()
    if not cli:
        if stream:
            def _g():
                for w in "Dev mode: placeholder stream.".split():
                    yield w + " "
            return _g()
        return "⚠️ OpenAI disabled on server."
    if stream:
        resp = cli.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.5, stream=True)
        def _gen():
            for ev in resp:
                piece = ev.choices[0].delta.content or ""
                if piece: yield piece
        return _gen()
    else:
        resp = cli.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.5)
        return resp.choices[0].message.content or ""

# -------- Faith detection ----------
FAITH_HINTS = {
    "bible":  ["bible","christ","jesus","psalm","proverb","philippians","isaiah","genesis","verse"],
    "quran":  ["quran","koran","allah","surah","ayat","ayah","muhammad","islam","hadith"],
    "talmud": ["talmud","mishnah","gemara","rabbi","bavli","tractate","halakha"],
}
FAITH_MAP = {
    "christian":"bible","christianity":"bible","bible":"bible","jesus":"bible","christ":"bible",
    "muslim":"quran","islam":"quran","quran":"quran","koran":"quran","surah":"quran","ayah":"quran",
    "jew":"talmud","jewish":"talmud","talmud":"talmud","mishnah":"talmud","gemara":"talmud",
}

def try_set_faith(message: str, s: SessionState) -> None:
    m = message.lower()
    for k,v in FAITH_MAP.items():
        if re.search(rf"\b{re.escape(k)}\b", m): s.faith = v; return
    for corp, words in FAITH_HINTS.items():
        if any(w in m for w in words):
            s.faith = s.faith or corp
            return

def detect_corpus(message: str, s: SessionState) -> str:
    if s.faith: return s.faith
    m = message.lower()
    scores = {corp: sum(w in m for w in words) for corp,words in FAITH_HINTS.items()}
    return max(scores, key=scores.get)

# -------- Memory helpers ----------
REMEMBER_RE = re.compile(r"\bremember\b", re.I)
IS_RE       = re.compile(r"^\s*(?P<k>.+?)\s+(?:is|are|=)\s+(?P<v>.+?)\s*$", re.I)

def normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(the|a|an|this|that)\s+", "", s)
    return re.sub(r"\s+"," ", s)

def parse_and_remember(message: str, s: SessionState) -> Optional[Dict[str,str]]:
    if not REMEMBER_RE.search(message): return None
    body = re.sub(r"^\s*remember\s*(that|:)?\s*", "", message, flags=re.I).strip()
    m = IS_RE.match(body)
    if m:
        k = normalize_key(m.group("k")); v = m.group("v").strip()
        if k: s.facts[k]=v; return {"key":k,"value":v}
    words = body.split()
    k = normalize_key(" ".join(words[:4])) or "note"
    v = " ".join(words[4:]) or body
    s.facts[k]=v
    return {"key":k,"value":v}

def lookup_memory_answer(message: str, s: SessionState) -> Optional[str]:
    q1 = re.compile(r"^\s*what\s+(color|colour)\s+is\s+(?:the\s+)?(?P<x>[^?!.]+)\??\s*$", re.I)
    q2 = re.compile(r"^\s*what\s+is\s+(?:the\s+)?(?P<x>[^?!.]+)\??\s*$", re.I)
    m = q1.match(message) or q2.match(message)
    if not m: return None
    x = normalize_key(m.group("x"))
    return s.facts.get(x) or s.facts.get(x.replace("the ","",1), None)

# -------- Hybrid search + MMR ----------
def hybrid_search(query: str, corpus_name: str, top_k: int = 8) -> List[Dict[str,Any]]:
    docs = CORPORA.get(corpus_name, [])
    if not docs: return []
    q_tokens = set(tokenize(query))
    lex_scores: List[Tuple[float,int]] = []
    for i, d in enumerate(docs):
        t = d.get("text","") or ""
        dt = set(tokenize(t))
        js = jaccard(q_tokens, dt)
        lex_scores.append((js, i))
    q_emb = embed_query(query)
    vec_scores: List[Tuple[float,int]] = []
    if q_emb:
        for i, d in enumerate(docs):
            e = d.get("embedding")
            s = cos(q_emb, e) if isinstance(e, list) else 0.0
            vec_scores.append((s, i))
    else:
        vec_scores = [(0.0, i) for i in range(len(docs))]
    def zscore(lst: List[Tuple[float,int]]) -> Dict[int,float]:
        vals = [x for x,_ in lst]
        if not vals: return {}
        mu = sum(vals)/len(vals)
        sd = (sum((x-mu)**2 for x in vals)/len(vals))**0.5 or 1.0
        return {j:(x-mu)/sd for x,j in lst}
    L = zscore(lex_scores); V = zscore(vec_scores); alpha = 0.6
    blended = [(alpha*L.get(i,0.0) + (1-alpha)*V.get(i,0.0), i) for i in range(len(docs))]
    blended.sort(reverse=True)
    candidates = [docs[i] for _,i in blended[: max(top_k*3, top_k+4)]]
    picked: List[Dict[str,Any]] = []; picked_refs = set(); lambda_div = 0.7
    while candidates and len(picked) < top_k:
        best, best_score = None, -1e9
        for d in candidates:
            base = 1.0
            ref = d.get("ref","")
            if ref and ref in picked_refs: base -= 0.5
            sim_to_sel = 0.0
            if picked:
                dt = set(tokenize(d.get("text","")))
                sim_to_sel = max(jaccard(dt, set(tokenize(p.get("text","")))) for p in picked)
            score = lambda_div*base - (1-lambda_div)*sim_to_sel + random.uniform(-0.01, 0.01)
            if score > best_score: best, best_score = d, score
        if not best: break
        picked.append(best)
        if best.get("ref"): picked_refs.add(best["ref"])
        candidates.remove(best)
    return picked[:top_k]

def choose_best_unique(s: SessionState, hits: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    if not hits: return None
    unseen = [h for h in hits if h.get("ref") and h["ref"] not in s.seen_refs]
    choice = (unseen or hits)[0]
    ref = choice.get("ref")
    if ref: s.seen_refs.add(ref)
    return choice

# -------- SSE utils ----------
def sse_event(event: Optional[str], data: str) -> str:
    if event:
        return f"event: {event}\n" + "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"
    return "\n".join(f"data: {line}" for line in data.splitlines()) + "\n\n"

# -------- App + CORS ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # explicit origin so cookies work with credentials
    allow_origins=["https://dturnover.github.io", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

@app.get("/")
def root(): return PlainTextResponse("ok")

@app.get("/health")
def health(): return PlainTextResponse("OK")

@app.get("/diag_rag")
def diag(): return {"ok": True, "corpora": corpus_counts()}

def build_system_prompt(s: SessionState) -> str:
    lines = [
        "You are Fight Chaplain: calm, concise, and encouraging.",
        "Offer practical corner-coach guidance with spiritual grounding.",
        "When you include a verse, quote it plainly and append [CIT: Book Chapter:Verse] if a ref exists.",
        "Keep answers tight.",
    ]
    if s.faith: lines.append(f"User faith preference: {s.faith}. Prefer that corpus.")
    if s.rollup: lines.append(f"SESSION_SUMMARY: {s.rollup}")
    if s.facts:
        lines.append("STICKY_NOTES:")
        for k,v in list(s.facts.items())[:30]:
            lines.append(f"- {k} = {v}")
    return "\n".join(lines)

def _build_messages(s: SessionState, user: str) -> List[Dict[str,str]]:
    msgs = [{"role":"system","content":build_system_prompt(s)}]
    msgs += s.history[-12:]
    msgs.append({"role":"user","content":user})
    return msgs

def attach_citation(text: str, hit: Optional[Dict[str,Any]]) -> str:
    if not hit: return text
    ref = (hit.get("ref") or "").strip()
    return f'{text} [CIT: {ref}]' if ref else text

def maybe_rollup(s: SessionState) -> None:
    if len(s.history) >= 24 and not s.rollup:
        user_lines = [m["content"] for m in s.history if m["role"]=="user"]
        if user_lines:
            s.rollup = "User themes: " + "; ".join(user_lines[-6:])[:500]

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()

    base = Response()
    sid, s = get_or_create_sid(request, base)
    remembered = parse_and_remember(message, s)
    try_set_faith(message, s)
    corpus = detect_corpus(message, s)

    mem = lookup_memory_answer(message, s)
    if mem:
        s.history.append({"role":"user","content":message})
        s.history.append({"role":"assistant","content":mem})
        return JSONResponse({"response": mem, "sid": sid, "sources": []}, headers=dict(base.headers))

    emotional = any(w in message.lower() for w in
                    ["scared","afraid","anxious","nervous","panic","hurt","down","lost",
                     "depressed","angry","worried","fight","injury","pain","tired","grateful"])
    asks_scripture = any(w in message.lower() for w in
                         ["verse","scripture","psalm","quote","recite","ayah","surah","mishnah","talmud","[cit"])
    rag_hit = None
    if emotional or asks_scripture:
        hits = hybrid_search(message, corpus, top_k=8)
        rag_hit = choose_best_unique(s, hits)

    if rag_hit:
        out = attach_citation(rag_hit.get("text",""), rag_hit)
        s.history.append({"role":"user","content":message})
        s.history.append({"role":"assistant","content":out})
        maybe_rollup(s)
        return JSONResponse(
            {"response": out, "sid": sid,
             "sources": [{"ref": rag_hit.get("ref",""), "source": rag_hit.get("source","")}]},
            headers=dict(base.headers),
        )

    out = call_openai(_build_messages(s, message), stream=False)
    reply = out if isinstance(out, str) else "".join(list(out))
    s.history.append({"role":"user","content":message})
    s.history.append({"role":"assistant","content":reply})
    maybe_rollup(s)
    return JSONResponse({"response": reply, "sid": sid, "sources": []}, headers=dict(base.headers))

@app.get("/chat_sse")
async def chat_sse(request: Request, q: str, sid: Optional[str] = None):
    base = Response()
    if sid: request.headers.__dict__.setdefault("_list", []).append((b"x-session-id", sid.encode()))
    sid2, s = get_or_create_sid(request, base)
    try_set_faith(q, s)
    corpus = detect_corpus(q, s)

    def gen() -> Generator[str, None, None]:
        yield ": connected\n\n"
        yield sse_event("ping", "hi")
        emotional = any(w in q.lower() for w in
                        ["scared","afraid","anxious","nervous","panic","hurt","down","lost",
                         "depressed","angry","worried","fight","injury","pain","tired","grateful"])
        asks_scripture = any(w in q.lower() for w in
                             ["verse","scripture","psalm","quote","recite","ayah","surah","mishnah","talmud","[cit"])
        rag_hit = None
        if emotional or asks_scripture:
            hits = hybrid_search(q, corpus, top_k=8)
            rag_hit = choose_best_unique(s, hits)
        if rag_hit:
            text = attach_citation(rag_hit.get("text",""), rag_hit)
            buf = ""
            for token in text.split():
                buf += token + " "
                if len(buf) >= 120 or token.endswith((".", "!", "?")):
                    yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
                    buf = ""
            if buf: yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
            s.history.append({"role":"user","content":q})
            s.history.append({"role":"assistant","content":text})
            maybe_rollup(s)
            yield sse_event("done", json.dumps({"sid": sid2}))
            return
        stream = call_openai(_build_messages(s, q), stream=True)  # type: ignore
        buf = ""
        for piece in stream:
            buf += piece
            if re.search(r"[.!?]\s+$", buf) or len(buf) >= 140:
                yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
                buf = ""
        if buf: yield f"data: {json.dumps({'text': buf.strip()})}\n\n"
        s.history.append({"role":"user","content":q})
        s.history.append({"role":"assistant","content":""})
        maybe_rollup(s)
        yield sse_event("done", json.dumps({"sid": sid2}))
    return StreamingResponse(gen(), media_type="text/event-stream", headers=dict(base.headers))

@app.get("/chat_sse_get")
def chat_sse_get():
    def gen():
        yield ": connected\n\n"
        yield sse_event(None, "This is a GET stream endpoint.\n")
    return StreamingResponse(gen(), media_type="text/event-stream")
