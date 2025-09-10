# backend/app.py
# Fight Chaplain — SSE streaming, session memory, HYBRID RAG, and strict
# "no-quote-without-RAG" policy to avoid hallucinated scripture.

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
            temperature=0.2,     # tighter to reduce drift
            stream=True
        )
        def _gen():
            for ev in resp:
                piece = ev.choices[0].delta.content or ""
                if piece:
                    yield piece
        return _gen()
    resp = cli.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, temperature=0.2
    )
    return resp.choices[0].message.content or ""

# ---------- Session ----------
class SessionState:
    def __init__(self) -> None:
        # history[0] is a placeholder; we compose a live system each turn
        self.history: List[Dict[str,str]] = [{"role":"system","content":""}]
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
                "book": o.get("book") or "",
                "chapter": o.get("chapter") or o.get("chap") or "",
                "verse": o.get("verse") or o.get("ayah") or "",
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
def tokenize(s: str) -> List[str]: return re.findall(r"[A-Za-z0-9]+", s.lower())

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
    if any(f" {w} " in m for w in ASK_WORDS): return True
    if any(f" {w} " in m for w in EMO_WORDS): return True
    return False

# ---------- Passage cleaning & pretty citation ----------
_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*")   # remove KJV editorial [words]
_WS_RE = re.compile(r"\s+")

def clean_passage(txt: str) -> str:
    """Remove editorial [brackets], collapse whitespace, trim."""
    if not txt: return ""
    txt = _BRACKET_RE.sub(" ", txt)
    txt = _WS_RE.sub(" ", txt)
    return txt.strip()

_BIBLE_REF_RE = re.compile(r'\b((?:[1-3]\s+)?[A-Za-z][A-Za-z ]{1,23})\s+(\d{1,3}):(\d{1,3})\b')

def _clean_book_name(name: str) -> str:
    name = re.sub(r'\s+', ' ', name.strip())
    parts = name.split(' ', 1)
    if len(parts) == 2 and parts[0] in {'1','2','3'}:
        return parts[0] + ' ' + parts[1].title()
    return name.title()

def extract_bible_ref_from_text(txt: str) -> str:
    m = _BIBLE_REF_RE.search(txt or "")
    if not m: return ""
    book, chap, verse = m.groups()
    return f"{_clean_book_name(book)} {int(chap)}:{int(verse)}"

def natural_ref(hit: Dict[str, Any]) -> str:
    src = (hit.get("source") or "").strip().lower()
    txt = hit.get("text") or ""
    if src == "bible":
        pretty = extract_bible_ref_from_text(txt)
        if pretty: return pretty
        label = "Bible"
    elif src == "quran":
        label = "Quran"
    elif src == "talmud":
        label = "Talmud"
    else:
        label = (hit.get("source") or "Source").title()
    ref = (hit.get("ref") or "").strip()
    for pref in ("bible-", "quran-", "talmud-"):
        if ref.lower().startswith(pref):
            ref = ref[len(pref):]
            break
    return f"{label} {ref}" if ref else label

# ---------- Hybrid search ----------
def hybrid_search(query: str, corpus_name: str, top_k: int = 6) -> List[Dict[str,Any]]:
    """
    Hybrid = lexical (Jaccard on tokens) + vector (cosine on embeddings) with z-score blend.
    """
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
        return []

    def zscore(lst: List[Tuple[float,int]]) -> Dict[int,float]:
        vals = [x for x,_ in lst]
        mu = sum(vals)/len(vals) if vals else 0.0
        sd = math.sqrt(sum((x-mu)**2 for x in vals)/len(vals)) or 1.0
        return {j:(x-mu)/sd for x,j in lst}

    L = zscore(lex_scores); V = zscore(vec_scores)
    alpha = 0.6
    blend = [(alpha*L.get(i,0.0) + (1-alpha)*V.get(i,0.0), i) for i in range(len(docs))]
    blend.sort(reverse=True)
    return [docs[i] for _, i in blend[:top_k]]

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

# ---------- 7-step flow + strict quote policy ----------
SYSTEM_BASE = """
You are Fight Chaplain — a calm, concise, spiritually grounded guide for combat sports athletes.
Unisex language only. Do NOT be a therapist or a generic chatbot.

ALWAYS FOLLOW THIS FLOW:
1) Begin by acknowledging the fighter’s emotional/spiritual state in warrior-aware language.
2) Briefly infer emotional tone + readiness (internally) and adapt your guidance.
3) Offer a short faith-based or reflective message (perseverance, failure, redemption).
4) Track conversation volume + emotional content.
5) If thresholds trip (keywords or sustained distress), escalate appropriately.
6) If need exceeds spiritual scope, suggest mental-health referral (e.g., BetterHelp). Otherwise, prefer referral to a faith leader.
7) Close with either a gentle next question OR an offer to connect them with a faith leader of their denomination.

CRITICAL QUOTE POLICY:
- You MUST NOT quote or cite any scripture unless explicit RETRIEVED PASSAGES are provided THIS TURN.
- If no retrieval is provided, speak in universal themes; invite the user to let you pull an exact passage.
- If the user’s faith is unknown, do not assume or name a tradition; ask gently.
- When retrieval IS provided, quote ONE short line verbatim and attribute naturally with an em dash (— Book C:V or tradition label).
""".strip()

CRISIS_KEYWORDS = {
    "suicide","kill myself","end it","self harm","hopeless","worthless","no way out",
    "panic attack","can’t breathe","cant breathe","severe anxiety","despair","give up"
}
DISTRESS_KEYWORDS = {
    "scared","afraid","panic","anxious","nervous","broken","alone","grief","grieving",
    "lost","depressed","overwhelmed","injury","pain","nightmare","doubt","shame","anger"
}

class _ChaplainState:
    __slots__ = ("turns","distress_hits","crisis_hits","escalate","faith_pref")
    def __init__(self) -> None:
        self.turns = 0
        self.distress_hits = 0
        self.crisis_hits = 0
        self.escalate = "none"  # "none" | "watch" | "refer-faith" | "refer-mental-health"
        self.faith_pref = ""

def _get_cs(s: SessionState) -> _ChaplainState:
    if not hasattr(s, "_chap"): setattr(s, "_chap", _ChaplainState())
    return getattr(s, "_chap")

def update_session_metrics(user_text: str, s: SessionState) -> None:
    cs = _get_cs(s)
    cs.turns += 1
    low = f" {user_text.lower()} "
    d = sum(1 for k in DISTRESS_KEYWORDS if f" {k} " in low)
    c = sum(1 for k in CRISIS_KEYWORDS   if f" {k} " in low)
    cs.distress_hits += d
    cs.crisis_hits   += c
    if cs.crisis_hits > 0:
        cs.escalate = "refer-mental-health"
    elif cs.distress_hits >= 3 or cs.turns >= 12:
        cs.escalate = "refer-faith"
    elif d > 0:
        cs.escalate = "watch"
    else:
        cs.escalate = "none"
    cs.faith_pref = (s.faith or "").title() if getattr(s, "faith", None) else ""

def system_message(s: SessionState, *, quote_allowed: bool, faith_known: bool) -> Dict[str,str]:
    base = SYSTEM_BASE
    if s.rollup:
        base += "\n\nSESSION_SUMMARY: " + s.rollup
    cs = _get_cs(s)
    status = {
        "turns": cs.turns,
        "distress_hits": cs.distress_hits,
        "crisis_hits": cs.crisis_hits,
        "escalate": cs.escalate,
        "faith_pref": cs.faith_pref or (s.faith or ""),
    }
    base += "\n\nSESSION_STATUS: " + json.dumps(status, ensure_ascii=False)
    base += f"\n\nTURN_LIMITS: quote_allowed={str(quote_allowed).lower()}, faith_known={str(faith_known).lower()}"
    base += (
        "\n\nCLOSER_POLICY: Always end with either a gentle next question "
        "OR an offer to connect them with a faith leader of their denomination."
    )
    return {"role":"system","content":base}

def apply_referral_footer(final_text: str, s: SessionState) -> str:
    cs = _get_cs(s)
    footer = ""
    if cs.escalate == "refer-mental-health":
        footer = (
            "\n\nIf you’re in immediate danger, contact local emergency services. "
            "For professional counseling, BetterHelp can connect you with a licensed counselor. "
            "I can also connect you with a faith leader if you’d like."
        )
    elif cs.escalate == "refer-faith":
        footer = (
            "\n\nIf you’d like, I can connect you with a faith leader from your denomination for personal support."
        )
    return (final_text or "").rstrip() + footer

ONBOARD_GREETING = (
    "Hello! It’s great to hear from you. How are you feeling today, both in and out of the ring? "
    "Do you practice within a specific faith tradition?"
)

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
        resp.headers["Access-Control-Methods"] = "GET, POST, OPTIONS"
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

# ---- Non-stream JSON chat ----
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    msg = (body.get("message") or "").strip()

    base = Response()
    sid, s = get_or_create_sid(request, base)

    # onboarding on first user turn
    if len(s.history) <= 1 and msg:
        s.history.append({"role":"assistant","content":ONBOARD_GREETING})

    try_set_faith(msg, s)
    update_session_metrics(msg, s)
    s.history.append({"role":"user","content":msg})
    if len(s.history) > 40:
        s.history = s.history[:1] + s.history[-39:]

    if wants_retrieval(msg):
        corp = detect_corpus(msg, s)
        hits = hybrid_search(msg, corp, top_k=6)
        if hits:
            top = hits[0]
            clean_text = clean_passage(top.get("text",""))
            pretty_ref = natural_ref(top)

            interleave_rule = (
                "QUOTE STYLE:\n"
                "- Use curly quotes around ONE short quoted line, then an em dash and the natural ref, e.g., "
                "“He maketh my feet like hinds’ feet.” — 2 Samuel 22:34\n"
                "- Surround the quote with 1–2 sentences of guidance BEFORE and AFTER. Do not quote twice."
            )
            ctx = (
                "RETRIEVED PASSAGES (you may quote EXACTLY ONE short line verbatim; then weave brief guidance around it):\n"
                f"- {pretty_ref} :: {clean_text}"
            )

            sys_msg = system_message(s, quote_allowed=True, faith_known=bool(s.faith))
            sys_msg["content"] += "\n\n" + interleave_rule + "\n" + ctx
            messages = [sys_msg] + s.history[1:]

            out = call_openai(messages, stream=False)
            reply = out if isinstance(out, str) else "".join(list(out))
            reply = apply_referral_footer(reply, s)
            s.history.append({"role":"assistant","content":reply})
            _maybe_rollup(s)
            return JSONResponse(
                {"response": reply, "sid": sid,
                 "sources":[{"ref": pretty_ref, "source": top.get("source","")}]},
                headers={"X-Session-Id": sid}
            )

    # no RAG → quotes forbidden this turn
    out = call_openai([system_message(s, quote_allowed=False, faith_known=bool(s.faith))] + s.history[1:], stream=False)
    reply = out if isinstance(out, str) else "".join(list(out))
    reply = apply_referral_footer(reply, s)
    s.history.append({"role":"assistant","content":reply})
    _maybe_rollup(s)
    return JSONResponse({"response": reply, "sid": sid, "sources": []}, headers={"X-Session-Id": sid})

# ---- Streaming SSE chat (no cookies used here) ----
@app.get("/chat_sse")
async def chat_sse(request: Request, q: str, sid: Optional[str] = None):
    if sid and sid in SESSIONS:
        sid2, s = sid, SESSIONS[sid]
    else:
        sid2 = sid or uuid.uuid4().hex
        s = SESSIONS.get(sid2) or SessionState()
        SESSIONS[sid2] = s

    if len(s.history) <= 1 and q:
        s.history.append({"role":"assistant","content":ONBOARD_GREETING})

    try_set_faith(q, s)
    update_session_metrics(q, s)
    origin = request.headers.get("origin")

    async def agen():
        yield b": connected\n\n"
        yield b"event: ping\ndata: hi\n\n"
        last_hb = asyncio.get_event_loop().time()

        def emit(txt: str) -> bytes:
            return ("data: " + json.dumps({"text": txt}) + "\n\n").encode("utf-8")

        try:
            # default: no quotes allowed unless RAG hits
            sys_msg = system_message(s, quote_allowed=False, faith_known=bool(s.faith))
            messages = [sys_msg] + s.history[1:] + [{"role":"user","content":q}]

            if wants_retrieval(q):
                corp = detect_corpus(q, s)
                hits = hybrid_search(q, corp, top_k=6)
                if hits:
                    top = hits[0]
                    clean_text = clean_passage(top.get("text",""))
                    pretty_ref = natural_ref(top)

                    interleave_rule = (
                        "QUOTE STYLE:\n"
                        "- Use curly quotes around ONE short quoted line, then an em dash and the natural ref, e.g., "
                        "“He maketh my feet like hinds’ feet.” — 2 Samuel 22:34\n"
                        "- Surround the quote with 1–2 sentences of guidance BEFORE and AFTER. Do not quote twice."
                    )
                    ctx = (
                        "RETRIEVED PASSAGES (you may quote EXACTLY ONE short line verbatim; then weave brief guidance around it):\n"
                        f"- {pretty_ref} :: {clean_text}"
                    )

                    sys_msg = system_message(s, quote_allowed=True, faith_known=bool(s.faith))
                    sys_msg["content"] += "\n\n" + interleave_rule + "\n" + ctx
                    messages = [sys_msg] + s.history[1:] + [{"role":"user","content":q}]

            # stream model guidance
            stream = call_openai(messages, stream=True)
            assistant_out, pending = "", ""
            for piece in stream:  # type: ignore
                if not piece: continue
                pending += piece
                if len(pending) >= 8 or "\n" in pending:
                    yield emit(pending); assistant_out += pending; pending = ""
                now = asyncio.get_event_loop().time()
                if (now - last_hb) > 5: yield b": hb\n\n"; last_hb = now
                await asyncio.sleep(0)
            if pending: yield emit(pending); assistant_out += pending

            # footer (if any)
            footer = apply_referral_footer("", s).strip()
            if footer: yield emit(footer)

            # persist
            stored = (assistant_out.strip() + ("\n\n" + footer if footer else "")).strip()
            s.history.append({"role":"user","content":q})
            s.history.append({"role":"assistant","content":stored})
            _maybe_rollup(s)
            yield ("event: done\ndata: " + json.dumps({"sid": sid2}) + "\n\n").encode("utf-8")

        except Exception as e:
            err = {"error": str(e)}
            yield ("event: error\ndata: " + json.dumps(err) + "\n\n").encode("utf-8")
            yield b": end\n\n"

    hdrs = {**sse_headers_for_origin(origin)}
    hdrs["Cache-Control"] = "no-cache"
    hdrs["X-Accel-Buffering"] = "no"
    return StreamingResponse(agen(), media_type="text/event-stream; charset=utf-8", headers=hdrs)

# ---- Infra sanity streams ----
@app.get("/sse_echo")
async def sse_echo():
    async def agen():
        i = 0
        while True:
            await asyncio.sleep(0.4); i += 1
            yield f"data: echo {i}\n\n".encode("utf-8")
            if i % 12 == 0: yield b": hb\n\n"
    return StreamingResponse(agen(), media_type="text/event-stream; charset=utf-8")

@app.get("/sse_test")
async def sse_test():
    async def agen():
        yield b": connected\n\n"
        for i in range(1, 6):
            await asyncio.sleep(0.4)
            yield f"data: tick {i}\n\n".encode("utf-8")
        yield b"event: done\ndata: ok\n\n"
    return StreamingResponse(agen(), media_type="text/event-stream; charset=utf-8")

@app.get("/chat_sse_get")
def chat_sse_get():
    def gen():
        yield ": connected\n\n"
        yield sse_event(None, "This is a GET stream endpoint.")
    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")
