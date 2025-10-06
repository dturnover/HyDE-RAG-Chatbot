# main.py
import json
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import config
import rag
import logic
import state  # ★★★ THE FIX IS HERE ★★★

app = FastAPI(title="Fight Chaplain Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type"],
)

@app.get("/diag")
async def diagnostics():
    return {"status": "ok", "openai_model": config.OPENAI_MODEL}

@app.post("/chat")
async def chat_handler(request: Request):
    try:
        data = await request.json()
        msg = data.get("message", "").strip()
        history = data.get("history", [])
        sid = data.get("sid")
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    if sid and sid in state.SESSIONS:
        s = state.SESSIONS[sid]
    else:
        sid = uuid.uuid4().hex
        s = logic.SessionState()
        state.SESSIONS[sid] = s
    
    s.history = history

    logic.try_set_faith(msg, s)
    logic.update_session_metrics(msg, s)
    
    current_conversation = s.history + [{"role": "user", "content": msg}]

    retrieval_ctx = logic.get_rag_context(msg, s)
    sys_msg = logic.system_message(s, quote_allowed=bool(retrieval_ctx), retrieval_ctx=retrieval_ctx)
    
    messages_for_llm = [sys_msg] + current_conversation

    async def stream_generator():
        yield f"event: start\ndata: {json.dumps({'sid': sid})}\n\n"
        
        full_response = ""
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
            
        final_response_with_footer = logic.apply_referral_footer(full_response, s._chap.escalate)
        footer = final_response_with_footer[len(full_response):]
        if footer:
             yield f"data: {json.dumps({'footer': footer})}\n\n"

        # Update the session's history with the full turn
        s.history.append({"role": "user", "content": msg})
        s.history.append({"role": "assistant", "content": final_response_with_footer})
        
        yield "data: [END]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")