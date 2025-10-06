# main.py
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import config
import rag
import logic

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
    return {"status": "ok", "openai_model": config.OPENAI_MODEL}

@app.post("/chat")
async def chat_handler(request: Request):
    try:
        data = await request.json()
        msg = data.get("message", "").strip()
        history = data.get("history", [])
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    
    # Create a temporary state object for this request
    s = logic.SessionState(history=history)
    
    # Scan the entire history to find the user's faith
    for turn in history:
        if turn.get("role") == "user":
            logic.try_set_faith(turn.get("content", ""), s)
    # Also check the current message for a faith keyword
    logic.try_set_faith(msg, s)
    
    # Prepare the full conversation for the LLM
    current_conversation = history + [{"role": "user", "content": msg}]

    retrieval_ctx = logic.get_rag_context(msg, s)
    sys_msg = logic.system_message(s, quote_allowed=bool(retrieval_ctx), retrieval_ctx=retrieval_ctx)
    
    messages_for_llm = [sys_msg] + current_conversation

    async def stream_generator():
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