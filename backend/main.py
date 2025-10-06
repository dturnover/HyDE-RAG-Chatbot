# main.py
import json, uuid
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import config, rag, logic, state

app = FastAPI(title="Fight Chaplain Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-Session-Id"], # Keep header for other potential uses
)

@app.get("/diag")
async def diagnostics():
    return { "status": "ok", "openai_model": config.OPENAI_MODEL }

# ★★★ MODIFIED CHAT HANDLER ★★★
@app.post("/chat")
async def chat_handler(request: Request):
    try:
        data = await request.json()
        msg = data.get("message", "").strip()
        sid = data.get("sid") # Look for the session ID in the JSON body
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    
    # Get or create session using the provided sid
    if sid and sid in state.SESSIONS:
        s = state.SESSIONS[sid]
    else:
        sid = uuid.uuid4().hex
        s = state.SessionState()
        state.SESSIONS[sid] = s

    # The rest of the logic flow remains the same
    logic.try_set_faith(msg, s)
    logic.update_session_metrics(msg, s)
    s.history.append({"role": "user", "content": msg})

    if len(s.history) > 20:
        s.history = s.history[:1] + s.history[-19:]

    retrieval_ctx = logic.get_rag_context(msg, s)
    sys_msg = logic.system_message(s, quote_allowed=bool(retrieval_ctx), retrieval_ctx=retrieval_ctx)
    messages_for_llm = [sys_msg] + s.history[1:]

    async def stream_generator():
        # First, send the session ID back to the client, just like the old app
        yield f"event: start\ndata: {json.dumps({'sid': sid})}\n\n"
        
        full_response = ""
        # ... the rest of the streaming logic is the same ...
        try:
            stream = rag.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages_for_llm,
                stream=True,
                temperature=0.2
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response += content
                    yield f"data: {json.dumps({'text': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
            
        final_response = logic.apply_referral_footer(full_response, s._chap.escalate)
        footer = final_response[len(full_response):]
        if footer:
             yield f"data: {json.dumps({'footer': footer})}\n\n"

        s.history.append({"role": "assistant", "content": final_response})
        yield "data: [END]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)