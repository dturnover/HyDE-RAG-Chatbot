# app.py — Responses-style conversational memory with streaming
import os, json, asyncio, uuid
from typing import Optional, List, Dict

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
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # << use gpt-5-mini

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

SESSIONS: Dict[str, Session] = {}

def get_sid(request: Request) -> str:
    sid = request.headers.get("x-session-id") or request.query_params.get("sid") or ""
    if not sid:
        sid = uuid.uuid4().hex
    if sid not in SESSIONS:
        SESSIONS[sid] = Session()
    return sid

def trim_history(sess: Session, keep:int=20):
    # keep latest N turns (+system at index 0)
    sys = sess.history[0:1]
    rest = sess.history[1:][-keep:]
    sess.history = sys + rest

# ---------------- OpenAI helpers ----------------
def openai_client() -> Optional["OpenAI"]:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def dev_stub_reply(prompt: str) -> str:
    return f"DEV: {prompt[:128]}"

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

    sess.history.append({"role":"user","content":user_msg})
    trim_history(sess)

    client = openai_client()
    if client is None:
        reply = dev_stub_reply(user_msg)
    else:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=sess.history,
            temperature=0.4,
        )
        reply = resp.choices[0].message.content or ""

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

# ---------------- /chat_sse (streaming with history) ----------------
@app.api_route("/chat_sse", methods=["GET","POST"])
async def chat_sse(request: Request, q: Optional[str] = None, sid: Optional[str] = None):
    # Accept sid via header or query
    header_sid = request.headers.get("x-session-id") or ""
    sid = sid or header_sid or get_sid(request)
    sess = SESSIONS[sid]

    # Get user message from GET ?q= or POST JSON {message}
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

        # Add user message to history first
        sess.history.append({"role":"user","content":user_msg})
        trim_history(sess)

        client = openai_client()
        # DEV stream if no OpenAI
        if client is None:
            acc = ""
            for tok in ["dev"," ","stream"," ","ok"]:
                yield sse_data({"text": tok})
                acc += tok
                await asyncio.sleep(0.1)
            # append full assistant reply to history
            sess.history.append({"role":"assistant","content":acc})
            trim_history(sess)
            yield sse_event("done", {"sid": sid})
            return

        # Real model stream
        stream = client.chat.completions.create(
            model=MODEL,
            messages=sess.history,
            temperature=0.4,
            stream=True,
        )
        acc = ""
        for ev in stream:
            delta = ev.choices[0].delta.content or ""
            if delta:
                acc += delta
                yield sse_data({"text": delta})
                await asyncio.sleep(0)  # allow flush
        # append full assistant reply for future memory
        sess.history.append({"role":"assistant","content":acc})
        trim_history(sess)
        yield sse_event("done", {"sid": sid})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Session-Id": sid,
    }
    return StreamingResponse(agen(), media_type="text/event-stream", headers=headers)
