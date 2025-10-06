# main.py
import json
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import config
import rag
import logic
import state  # ★★★ CRITICAL FIX IS HERE ★★★

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
    print("\n--- [DEBUG] DIAGNOSTIC ENDPOINT HIT ---")
    return { "status": "ok", "openai_model": config.OPENAI_MODEL }

@app.post("/chat")
async def chat_handler(request: Request):
    print("\n\n--- ★★★ [DEBUG] NEW REQUEST RECEIVED AT /chat ★★★ ---")
    try:
        data = await request.json()
        print(f"[DEBUG] Raw request body received: {data}")
        msg = data.get("message", "").strip()
        history = data.get("history", []) # Get history from client
        sid = data.get("sid") # Get sid from client
        if not msg:
            print("[DEBUG] ERROR: Message is empty. Raising HTTPException.")
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception as e:
        print(f"[DEBUG] ERROR: Failed to parse JSON body. Error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    
    print(f"[DEBUG] Parsed SID from request: {sid}")
    
    if sid and sid in state.SESSIONS:
        print(f"[DEBUG] Found existing session for SID: {sid}")
        s = state.SESSIONS[sid]
    else:
        sid = uuid.uuid4().hex
        print(f"[DEBUG] No session found. Creating new session with SID: {sid}")
        s = logic.SessionState() # Create instance from logic.py
        state.SESSIONS[sid] = s

    print(f"[DEBUG] Syncing state with client history of length: {len(history)}")
    s.history = history

    print("[DEBUG] --- Calling logic.try_set_faith ---")
    logic.try_set_faith(msg, s)
    print(f"[DEBUG] --- After try_set_faith, session faith is: {s.faith} ---")

    print("[DEBUG] --- Calling logic.update_session_metrics ---")
    logic.update_session_metrics(msg, s)

    current_conversation = s.history + [{"role": "user", "content": msg}]

    print("[DEBUG] --- Calling logic.get_rag_context ---")
    retrieval_ctx = logic.get_rag_context(msg, s)
    print(f"[DEBUG] --- After get_rag_context, retrieval_ctx is: {'Present' if retrieval_ctx else 'None'} ---")

    print("[DEBUG] --- Calling logic.system_message ---")
    sys_msg = logic.system_message(s, quote_allowed=bool(retrieval_ctx), retrieval_ctx=retrieval_ctx)
    
    messages_for_llm = [sys_msg] + current_conversation
    print(f"[DEBUG] Final number of messages being sent to LLM: {len(messages_for_llm)}")
    print(f"[DEBUG] System Message Content Snapshot: {sys_msg['content'][:200]}...")

    async def stream_generator():
        print("[DEBUG] Stream generator started.")
        yield f"event: start\ndata: {json.dumps({'sid': sid})}\n\n"
        print("[DEBUG]   - Sent 'start' event with SID.")
        
        full_response = ""
        try:
            print("[DEBUG]   - Calling OpenAI API...")
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
            print("[DEBUG]   - OpenAI stream finished.")
        except Exception as e:
            print(f"[DEBUG]   - ERROR during OpenAI stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
            
        final_response_with_footer = logic.apply_referral_footer(full_response, s._chap.escalate)
        footer = final_response_with_footer[len(full_response):]
        if footer:
             yield f"data: {json.dumps({'footer': footer})}\n\n"

        # Update the session's history with the full turn
        current_conversation.append({"role": "assistant", "content": final_response_with_footer})
        s.history = current_conversation
        
        yield "data: [END]\n\n"
        print("[DEBUG]   - Sent 'END' event. Stream finished.")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")