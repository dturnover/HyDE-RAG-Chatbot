# main.py
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import config, rag, logic # No longer imports state

app = FastAPI(title="Fight Chaplain Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/diag")
async def diagnostics():
    return { "status": "ok", "openai_model": config.OPENAI_MODEL }

@app.post("/chat")
async def chat_handler(request: Request):
    try:
        data = await request.json()
        msg = data.get("message", "").strip()
        # Get the entire history from the client
        history = data.get("history", [])
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    
    # Create a temporary, single-use session object for this turn
    s = logic.SessionState(history=history)
    
    logic.try_set_faith(msg, s)
    logic.update_session_metrics(msg, s)
    
    # Use the history provided by the client
    current_conversation = s.history + [{"role": "user", "content": msg}]

    retrieval_ctx = logic.get_rag_context(msg, s)
    sys_msg = logic.system_message(s, quote_allowed=bool(retrieval_ctx), retrieval_ctx=retrieval_ctx)
    
    messages_for_llm = [sys_msg] + current_conversation

    async def stream_generator():
        # ... the streaming logic remains the same ...
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
                    yield f"data: {json.dumps({'text': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
            
        yield "data: [END]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# Note: We are now defining SessionState inside logic.py, so state.py is no longer needed.
# Ensure you have this class definition in your logic.py file.