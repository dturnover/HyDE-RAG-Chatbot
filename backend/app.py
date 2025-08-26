# app.py — Responses-style memory + auto-summary rollup + streaming
import os, json, asyncio, uuid
from typing import Optional, List, Dict, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse

# --- OpenAI SDK (graceful if missing) ---
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Default to a widely available model; override via env OPENAI_MODEL
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Memory rollup tuning
KEEP_TURNS = 20          # keep last N turns verbatim (plus system at index 0)
ROLLUP_MAX_CHARS = 1500  # cap for persistent memory blob

app = FastAPI()

# CORS (no credentials or cookies)
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

# ---------------- Health ----------------
@app.get("/health")
def health():
    return PlainTextResponse("OK")

@app.get("/")
def root():
    return PlainTextResponse("ok")

# ---------------- Session store ----------------
class Session:
    def __init__(self) -> None:
        # Chat-style history: [{"role":"system|user|assistant","content":"..."}]
        self.history: List[Dict[str,str]] = [
            {"role":"system","content":
             "You are Fight Chaplain: calm, concise, encouraging. "
             "Offer practical corner-coach guidance with spiritual grounding. "
             "Keep answers tight; avoid filler."
            }
        ]
        # Persistent rolled-up memory of older turns
        self.memory_blob: str = ""

SESSIONS: Dict[str, Session] = {}

def get_sid(request: Request) -> str:
    sid = request.headers.get("x-session-id") or request.query_params.get("sid") or ""
    if not sid:
        sid = uuid.uuid4().hex
    if sid not in SESSIONS:
        SESSIONS[sid] = Session()
    return sid

def trim_history(sess: Session, keep:int=KEEP_TURNS):
    # keep latest N turns (+system at index 0)
    sys = sess.history[0:1]
    rest = sess.history[1:][-keep:]
    sess.history = sys + rest

def openai_client() -> Optional["OpenAI"]:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def dev_stub_reply(prompt: str) -> str:
    return f"DEV: {prompt[:128]}"

def _system_with_memory(sess: Session) -> Dict[str,str]:
    # Combine original system + PERSISTENT_MEMORY (if any)
    base = sess.history[0]["content"]
    if sess.memory_blob:
        base = (base + "\n\nPERSISTENT_MEMORY (facts & context from earlier turns; prefer these when relevant):\n"
                + sess.memory_blob.strip())
    return {"role":"system","content": base}

# ---------------- Rollup (auto-summary) ----------------
def _needs_rollup(sess: Session) -> bool:
    # roll up if more than KEEP_TURNS user/assistant turns beyond system
    return len(sess.history) > (KEEP_TURNS + 1)

def _split_for_rollup(sess: Session) -> Tuple[List[Dict[str,str]], List[Dict[str,str]]]:
    """Return (older, recent) excluding system; caller reinserts system."""
    older = sess.history[1:-KEEP_TURNS] if len(sess.history) - 1 > KEEP_TURNS else []
    recent = sess.history[-KEEP_TURNS:] if len(sess.history) - 1 >= KEEP_TURNS else sess.history[1:]
    return older, recent

def _merge_blobs(existing: str, new: str, cap: int = ROLLUP_MAX_CHARS) -> str:
    merged = (existing + "\n" + new).strip() if existing else new.strip()
    # simple cap: keep last cap chars (most recent rollups)
    if len(merged) > cap:
        merged = merged[-cap:]
    return merged

def _summarize_old_turns(older: List[Dict[str,str]]) -> str:
    """Summarize older turns into a compact, factual blob."""
    if not older: return ""
    client = openai_client()
    # Build a tiny prompt describing the format we want.
    sys = {"role":"system","content":
           "Summarize the chat turns into a compact, factual memory blob. "
           "Keep it under ~12 bullets when possible. Prioritize: name, faith/spiritual tradition, goals, "
           "preferences (tone, brevity), important facts (e.g., 'penguin is painted blue'), and unresolved tasks. "
           "Be terse and concrete; no narrative fluff."}
    # Provide older turns directly
    messages = [sys] + older + [{"role":"user","content":"Return only the bullet list. No preamble."}]
    if client is None:
        # DEV fallback: naive last-lines concat
        text = "\n".join(f"- {m['role']}: {m['content'][:120]}" for m in older[-12:])
        return text[:ROLLUP_MAX_CHARS]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    out = resp.choices[0].message.content or ""
    # light sanitize
    out = out.strip()
    if len(out) > ROLLUP_MAX_CHARS:
        out = out[:ROLLUP_MAX_CHARS]
    return out

def rollup_if_needed(sess: Session) -> None:
    if not _needs_rollup(sess):
        return
    older, recent = _split_for_rollup(sess)
    if not older:
        return
    blob = _summarize_old_turns(older)
    if blob:
        sess.memory_blob = _merge_blobs(sess.memory_blob, blob, ROLLUP_MAX_CHARS)
    # rebuild history: system (with same base) + recent
    base_system = sess.history[0]  # keep original system text (without blob; we inject blob per request)
    sess.history = [base_system] + recent

# ---------------- /chat (non-stream) ----------------
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_msg = (body.get("message") or "").strip()

    sid = get_sid(request)
    sess = SESSIONS[sid]

    if user_msg.lower() == "ping":
        return JSONResponse({"response":"pong", "sid": sid}, headers={"X-Session-Id": sid})

    # Add user turn
    sess.history.append({"role":"user","content":user_msg})
    # Roll up if needed before we call the model
    rollup_if_needed(sess)
    trim_history(sess)

    client = openai_client()
    sys_msg = _system_with_memory(sess)
    messages = [sys_msg] + sess.history[1:]  # replace system with the combined one

    if client is None:
        reply = dev_stub_reply(user_msg)
    else:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
        )
        reply = resp.choices[0].message.content or ""

    # Add assistant turn
    sess.history.append({"role":"assistant","content":reply})
    trim_history(sess)

    return JSONResponse({"response": reply, "sid": sid}, headers={"X-Session-Id": sid})

# ---------------- SSE helpers ----------------
def sse_data(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

def sse_event(name: str, data: dict | str = "") -> str:
    out = f"event: {name}\n"
    if data != "":
        if isinstance(data, dict):
            out += f"data: {json.dumps(data, ensure_ascii=False)}\n"
        else:
            for ln in str(data).splitlines():
                out += f"data: {ln}\n"
    return out + "\n"

# ---------------- /chat_sse (streaming with history + rollup) ----------------
@app.api_route("/chat_sse", methods=["GET","POST"])
async def chat_sse(request: Request, q: Optional[str] = None, sid: Optional[str] = None):
    header_sid = request.headers.get("x-session-id") or ""
    sid = sid or header_sid or get_sid(request)
    sess = SESSIONS[sid]

    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        user_msg = (body.get("message") or "").strip()
    else:
        user_msg = (q or "").strip()

    async def agen():
        yield ": connected\n\n"
        yield sse_event("ping", "hi")

        # Add user turn and roll up if needed
        sess.history.append({"role":"user","content":user_msg})
        rollup_if_needed(sess)
        trim_history(sess)

        client = openai_client()
        sys_msg = _system_with_memory(sess)
        messages = [sys_msg] + sess.history[1:]

        # DEV stream
        if client is None:
            acc = ""
            for tok in ["dev"," ","stream"," ","ok"]:
                yield sse_data({"text": tok}); acc += tok
                await asyncio.sleep(0.1)
            sess.history.append({"role":"assistant","content":acc})
            trim_history(sess)
            yield sse_event("done", {"sid": sid})
            return

        # Real model stream
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            stream=True,
        )
        acc = ""
        for ev in stream:
            delta = ev.choices[0].delta.content or ""
            if delta:
                acc += delta
                yield sse_data({"text": delta})
                await asyncio.sleep(0)
        sess.history.append({"role":"assistant","content":acc})
        trim_history(sess)
        yield sse_event("done", {"sid": sid})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Session-Id": sid,
    }
    return StreamingResponse(agen(), media_type="text/event-stream", headers=headers)
