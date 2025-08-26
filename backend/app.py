# app.py — memory + rollup + streaming + Phase 6 vector retrieval
import os, json, asyncio, uuid, re
from typing import Optional, List, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse

# OpenAI
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---- RAG (Phase 6) ----
from rag import STORE, init_store, EMBED_MODEL

# memory settings
KEEP_TURNS = 20
ROLLUP_MAX_CHARS = 1500

app = FastAPI()

ALLOWED_ORIGINS = {
    "https://dturnover.github.io",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5500", "http://127.0.0.1:5500",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ---- health ----
@app.get("/health")
def health(): return PlainTextResponse("OK")
@app.get("/")
def root(): return PlainTextResponse("ok")

# ---- sessions ----
class Session:
    def __init__(self) -> None:
        self.history: List[Dict[str,str]] = [{
            "role":"system",
            "content":(
                "You are Fight Chaplain: calm, concise, encouraging. "
                "Offer practical corner-coach guidance with spiritual grounding. "
                "Keep answers tight; avoid filler. When I provide RETRIEVED PASSAGES, "
                "quote only from them if you choose to cite scripture and include the given citation tag."
            )
        }]
        self.memory_blob: str = ""

SESSIONS: Dict[str, Session] = {}

def get_sid(request: Request) -> str:
    sid = request.headers.get("x-session-id") or request.query_params.get("sid") or ""
    if not sid: sid = uuid.uuid4().hex
    if sid not in SESSIONS: SESSIONS[sid] = Session()
    return sid

def trim_history(sess: Session):
    sys = sess.history[0:1]
    rest = sess.history[1:][-KEEP_TURNS:]
    sess.history = sys + rest

def openai_client() -> Optional["OpenAI"]:
    if not (_HAS_OPENAI and OPENAI_API_KEY): return None
    return OpenAI(api_key=OPENAI_API_KEY)

def dev_stub_reply(prompt: str) -> str:
    return f"DEV: {prompt[:128]}"

def _system_with_memory(sess: Session) -> Dict[str,str]:
    base = sess.history[0]["content"]
    if sess.memory_blob:
        base = base + "\n\nPERSISTENT_MEMORY:\n" + sess.memory_blob.strip()
    return {"role":"system","content": base}

# ---- rollup ----
def _needs_rollup(sess: Session) -> bool: return len(sess.history) > (KEEP_TURNS + 1)

def _split_for_rollup(sess: Session):
    older = sess.history[1:-KEEP_TURNS] if len(sess.history)-1 > KEEP_TURNS else []
    recent = sess.history[-KEEP_TURNS:] if len(sess.history)-1 >= KEEP_TURNS else sess.history[1:]
    return older, recent

def _summarize_old_turns(client, older: List[Dict[str,str]]) -> str:
    if not older: return ""
    if client is None:
        txt = "\n".join(f"- {m['role']}: {m['content'][:120]}" for m in older[-12:])
        return txt[:ROLLUP_MAX_CHARS]
    sys = {"role":"system","content":
           "Summarize prior turns into bullet facts (name, faith, prefs, key facts, TODOs). Terse, <=12 bullets."}
    msgs = [sys] + older + [{"role":"user","content":"Return only the bullet list."}]
    r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2)
    out = (r.choices[0].message.content or "").strip()
    return out[:ROLLUP_MAX_CHARS]

def rollup_if_needed(client, sess: Session):
    if not _needs_rollup(sess): return
    older, recent = _split_for_rollup(sess)
    blob = _summarize_old_turns(client, older)
    if blob:
        merged = (sess.memory_blob + "\n" + blob).strip() if sess.memory_blob else blob
        if len(merged) > ROLLUP_MAX_CHARS: merged = merged[-ROLLUP_MAX_CHARS:]
        sess.memory_blob = merged
    base = sess.history[0]
    sess.history = [base] + recent

# ---- RAG triggers ----
EMO_WORDS = ["scared","afraid","anxious","nervous","panic","injury","hurt","grief","loss","fear","doubt","alone"]
ASK_WORDS = ["verse","scripture","ayah","surah","psalm","quote","passage","talmud","tractate","bible","quran"]

def wants_retrieval(msg: str) -> bool:
    m = msg.lower()
    return any(w in m for w in ASK_WORDS) or ("share" in m and ("verse" in m or "scripture" in m)) or any(w in m for w in EMO_WORDS)

def build_retrieval_context(sources: List[Dict], max_chars: int = 1200) -> str:
    if not sources: return ""
    lines = ["RETRIEVED PASSAGES (use if relevant; cite with [CIT:<id>]):"]
    used = 0
    for s in sources[:2]:  # keep it tight
        head = f"- [{s['trad']}] [CIT:{s['id']}] {s.get('ref','')}".strip()
        txt = s["text"].strip().replace("\n"," ")
        chunk = f"{head}\n  {txt}"
        if used + len(chunk) > max_chars: break
        lines.append(chunk)
        used += len(chunk)
    return "\n".join(lines)

# ---- endpoints ----
@app.post("/chat")
async def chat(request: Request):
    try: body = await request.json()
    except Exception: body = {}
    user_msg = (body.get("message") or "").strip()

    sid = get_sid(request)
    sess = SESSIONS[sid]
    client = openai_client()

    # user turn
    sess.history.append({"role":"user","content":user_msg})
    rollup_if_needed(client, sess)
    trim_history(sess)

    # optional retrieval
    sources = []
    if wants_retrieval(user_msg):
        init_store()  # idempotent; fast after first load
        sources = STORE.retrieve(client, query=user_msg, hint=user_msg, top_k_each=4, limit=6)
        if sources:
            ctx = build_retrieval_context(sources)
            if ctx:
                # inject as system preface for THIS call only
                sys_msg = _system_with_memory(sess)
                sys_msg["content"] = sys_msg["content"] + "\n\n" + ctx
                messages = [sys_msg] + sess.history[1:]
            else:
                messages = [_system_with_memory(sess)] + sess.history[1:]
        else:
            messages = [_system_with_memory(sess)] + sess.history[1:]
    else:
        messages = [_system_with_memory(sess)] + sess.history[1:]

    # model call
    if client is None:
        reply = dev_stub_reply(user_msg)
    else:
        r = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.4)
        reply = r.choices[0].message.content or ""

    sess.history.append({"role":"assistant","content":reply})
    trim_history(sess)
    return JSONResponse({"response": reply, "sid": sid, "sources": sources}, headers={"X-Session-Id": sid})

def sse_data(obj) -> str: return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
def sse_event(name: str, data: dict | str = "") -> str:
    out = f"event: {name}\n"
    if data != "":
        if isinstance(data, dict): out += f"data: {json.dumps(data, ensure_ascii=False)}\n"
        else:
            for ln in str(data).splitlines(): out += f"data: {ln}\n"
    return out + "\n"

@app.api_route("/chat_sse", methods=["GET","POST"])
async def chat_sse(request: Request, q: Optional[str] = None, sid: Optional[str] = None):
    header_sid = request.headers.get("x-session-id") or ""
    sid = sid or header_sid or get_sid(request)
    sess = SESSIONS[sid]
    client = openai_client()

    if request.method == "POST":
        try: body = await request.json()
        except Exception: body = {}
        user_msg = (body.get("message") or "").strip()
    else:
        user_msg = (q or "").strip()

    async def agen():
        yield ": connected\n\n"
        yield sse_event("ping", "hi")

        # user turn + rollup
        sess.history.append({"role":"user","content":user_msg})
        rollup_if_needed(client, sess)
        trim_history(sess)

        # maybe retrieve
        sources = []
        sys_msg = _system_with_memory(sess)
        if wants_retrieval(user_msg):
            init_store()
            sources = STORE.retrieve(client, query=user_msg, hint=user_msg, top_k_each=4, limit=6)
            if sources:
                ctx = build_retrieval_context(sources)
                if ctx:
                    sys_msg = {"role":"system","content": sys_msg["content"] + "\n\n" + ctx}

        messages = [sys_msg] + sess.history[1:]

        # dev stream
        if client is None:
            acc = ""
            for tok in ["dev"," ","stream"," ","ok"]:
                yield sse_data({"text": tok}); acc += tok
                await asyncio.sleep(0.05)
            sess.history.append({"role":"assistant","content":acc})
            trim_history(sess)
            if sources: yield sse_event("sources", {"sources": sources})
            yield sse_event("done", {"sid": sid})
            return

        # model stream
        stream = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.4, stream=True)
        acc = ""
        for ev in stream:
            delta = ev.choices[0].delta.content or ""
            if delta:
                acc += delta
                yield sse_data({"text": delta})
                await asyncio.sleep(0)
        sess.history.append({"role":"assistant","content":acc})
        trim_history(sess)
        if sources: yield sse_event("sources", {"sources": sources})
        yield sse_event("done", {"sid": sid})

    headers = {"Cache-Control": "no-cache","X-Accel-Buffering": "no","X-Session-Id": sid}
    return StreamingResponse(agen(), media_type="text/event-stream", headers=headers)

# initialize vector store on boot (non-fatal if files missing)
try:
    init_store()
except Exception:
    pass
