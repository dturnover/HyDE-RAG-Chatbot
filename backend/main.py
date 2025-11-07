# main.py
"""
This file is the main "server" for our chatbot. It uses the FastAPI
framework to create a web server that listens for and responds to
HTTP requests. This is the entry point for the entire application.
"""

import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import our other Python files
import config
import rag
import logic

# Initialize the FastAPI app
app = FastAPI(title="Fight Chaplain Backend")

@app.on_event("startup")
async def on_startup():
    """
    This function runs *once* when the FastAPI server starts.
    We use it to pre-calculate and load the crisis phrase
    embeddings (from logic.py) into memory. This way, the server
    is ready for instant crisis checking on the very first message.
    """
    logging.info("Server is starting up...")
    logic.initialize_crisis_embeddings()

# Add "CORS" middleware. This is a security setting that allows
# our web frontend (running on a different domain) to make
# requests to this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/diag")
async def diagnostics():
    """
    A "diagnostics" or "health check" endpoint.
    You can visit this URL (e.g., http://127.0.0.1:8000/diag) in a
    browser to see if the server is running and if it successfully
    connected to OpenAI and Pinecone.
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
    This is the main endpoint that handles all chat messages.
    It's the most important part of this file.
    """
    try:
        # Get the JSON data sent by the frontend
        data = await request.json()
        msg = data.get("message", "").strip()
        history = data.get("history", [])
        
        # Basic validation
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # --- This is where we run our core chatbot logic ---
    
    # 1. Create a new, blank "SessionState" object for this specific request
    s = logic.SessionState(history=history)
    
    # 2. Re-build faith state from history:
    #    Loop through all *past* messages to find the *last* faith mentioned.
    for turn in history:
        if turn.get("role") == "user":
            logic.try_set_faith(turn.get("content", ""), s)
            # ★★★ BUG FIX NOTE ★★★
            # There is purposefully NO 'break' statement here.
            # We *must* process the *entire* history to ensure
            # the faith is set to the *last* one the user mentioned.
    
    # 3. Now, check the *new* message for a faith update.
    logic.try_set_faith(msg, s)
    
    # 4. Run our 2-layer crisis check on the new message
    logic.update_session_state(msg, s)
    
    # 5. Create the full conversation to send to the AI
    current_conversation = history + [{"role": "user", "content": msg}]

    # 6. Get RAG context (if needed)
    #    This runs the HyDE logic and searches Pinecone.
    #    'retrieval_ctx' will be None or a "RETRIEVED PASSAGE: ..." string.
    retrieval_ctx = logic.get_rag_context(msg, s)
    
    # 7. Build the main "system message"
    #    This combines the base prompt, RAG rules, and the context we just got.
    sys_msg = logic.system_message(
        s, 
        quote_allowed=bool(retrieval_ctx), 
        retrieval_ctx=retrieval_ctx
    )
    
    # 8. Combine the instructions and the conversation
    messages_for_llm = [sys_msg] + current_conversation

    async def stream_generator():
        """
        This function contacts the OpenAI API and "yields" (sends)
        each piece of the response back to the frontend as it
        receives it. This creates the "typing" effect.
        """
        final_full_response = ""
        
        try:
            # Call the OpenAI API in streaming mode
            stream = rag.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages_for_llm,
                stream=True,
                temperature=0.2  # Low temperature for more focused answers
            )
            
            # Send each "chunk" of text as it arrives
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    final_full_response += content
                    # Send the chunk back to the frontend
                    yield f"data: {json.dumps({'text': content})}\n\n"
        
        except Exception as e:
            error_msg = f"Stream Error: {str(e)}"
            logging.error(error_msg, exc_info=True)
            # Send an error message to the frontend
            yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"
            return
            
        # --- After the stream is finished ---
        
        # 1. Check if we need to add a crisis/referral footer
        footer_text = logic.apply_referral_footer(final_full_response, s)
        
        # 2. If a footer exists, send it as a new, separate message
        if footer_text:
            yield f"data: {json.dumps({'text': footer_text})}\n\n"
        
        # 3. Send the "END" signal to tell the frontend we're done
        yield "data: [END]\n\n"

    # Return the stream generator to FastAPI, which handles the streaming response
    return StreamingResponse(stream_generator(), media_type="text/event-stream")