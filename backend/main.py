# main.py
# Ensures update_session_state and apply_referral_footer are called.
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging # Added for error logging

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
    # Check if clients are initialized
    clients_status = {
        "openai_client": "Initialized" if rag.client else "NOT Initialized",
        "pinecone_index": "Initialized" if rag.index else "NOT Initialized",
        "model": config.OPENAI_MODEL
    }
    return {"status": "ok", "clients": clients_status}

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
    
    # --- Client Logic Integration ---
    # 1. Scan history and current message for explicit faith
    for turn in history:
        if turn.get("role") == "user":
            logic.try_set_faith(turn.get("content", ""), s)
            if s.faith and s.faith != "bible_nrsv": break # Stop if non-default found
    
    # 2. Check current message for explicit faith (or set default)
    logic.try_set_faith(msg, s)
    
    # 3. Update escalation state
    logic.update_session_state(msg, s)
    
    # 4. Prepare conversation
    current_conversation = history + [{"role": "user", "content": msg}]

    # 5. Get RAG context (if needed)
    retrieval_ctx = logic.get_rag_context(msg, s)
    
    # 6. Generate system message
    sys_msg = logic.system_message(s, quote_allowed=bool(retrieval_ctx), retrieval_ctx=retrieval_ctx)
    
    messages_for_llm = [sys_msg] + current_conversation

    # Log the final system prompt being sent (for debugging)
    print(f"--- DEBUG [main.py] System Prompt ---")
    print(sys_msg['content'])
    print("---------------------------------")

    async def stream_generator():
        final_full_response = "" # To check for footer logic
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
                    final_full_response += content
                    yield f"data: {json.dumps({'text': content})}\n\n"
        
        except Exception as e:
            error_msg = f"Stream Error: {str(e)}"
            logging.error(error_msg, exc_info=True) # Log full traceback
            yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"
            return
            
        # Call apply_referral_footer *after* stream (Step 6 & 7)
        footer_text = logic.apply_referral_footer(final_full_response, s)
        if footer_text: # footer_text will start with \n\n if not empty
             yield f"data: {json.dumps({'text': footer_text})}\n\n" # Send footer as a new chunk
        
        yield "data: [END]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

