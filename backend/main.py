# main.py
#
# This file is the main "server" for our chatbot.
# The ONLY change is removing one "break" statement
# to fix the faith "glitch" bug.

import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

import config
import rag
import logic

app = FastAPI(title="Fight Chaplain Backend")

@app.on_event("startup")
async def on_startup():
    """
    This function runs *once* when the FastAPI server starts.
    We use it to pre-calculate and load the crisis phrase
    embeddings into memory so they are ready for instant use.
    """
    logging.info("Server is starting up...")
    logic.initialize_crisis_embeddings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/diag")
async def diagnostics():
    """
    A "diagnostics" or "health check" page.
    """
    clients_status = {
        "openai_client": "Initialized" if rag.client else "NOT Initialized",
        "pinecone_index": "Initialized" if rag.index else "NOT Initialized",
        "model": config.OPENAI_MODEL
    }
    return {"status": "ok", "clients": clients_status}

@app.post("/chat")
async def chat_handler(request: Request):
    """
    This is the main endpoint that handles chat messages.
    """
    try:
        data = await request.json()
        msg = data.get("message", "").strip()
        history = data.get("history", [])
        
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    s = logic.SessionState(history=history)
    
    # --- This is where we run our core chatbot logic ---
    
    # 1. Look through the old messages to find the *last* faith mentioned
    for turn in history:
        if turn.get("role") == "user":
            logic.try_set_faith(turn.get("content", ""), s)
            # ★★★ BUG FIX ★★★
            # The 'break' statement was removed from here.
            # This ensures the loop processes the *entire* history,
            # so the *last* faith mentioned is the one that sticks.
    
    # 2. Now, check the *new* message for a faith.
    logic.try_set_faith(msg, s)
    
    # 3. Check for crisis state
    logic.update_session_state(msg, s)
    
    # 4. Create the full conversation to send to the AI
    current_conversation = history + [{"role": "user", "content": msg}]

    # 5. Get RAG context (if needed)
    retrieval_ctx = logic.get_rag_context(msg, s)
    
    # 6. Build the main "system message"
    sys_msg = logic.system_message(
        s, 
        quote_allowed=bool(retrieval_ctx), 
        retrieval_ctx=retrieval_ctx
    )
    
    # 7. Combine the instructions and the conversation
    messages_for_llm = [sys_msg] + current_conversation

    async def stream_generator():
        """
        This function contacts the OpenAI API and yields (sends)
        each piece of the response as it receives it.
        """
        final_full_response = ""
        
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
            logging.error(error_msg, exc_info=True)
            yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"
            return
            
        # --- After the stream is finished ---
        footer_text = logic.apply_referral_footer(final_full_response, s)
        
        if footer_text:
            yield f"data: {json.dumps({'text': footer_text})}\n\n"
        
        yield "data: [END]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")