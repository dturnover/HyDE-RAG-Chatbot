# main.py
#
# This file is the main "server" for our chatbot.
# It uses FastAPI, a tool for building web APIs, to listen for
# messages from the user's web browser. It handles the "chat"
# requests, uses our other Python files (logic, rag) to figure out
# a response, and then streams that response back to the user.

import json  # Used to format the data we send back to the browser
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse  # This lets us send the response back piece-by-piece
from fastapi.middleware.cors import CORSMiddleware  # This lets a web browser on another domain talk to our server
import logging  # Used for logging any real errors that happen

# Import our own Python code
import config  # For settings
import rag  # For finding scripture (Retrieval-Augmented Generation)
import logic  # For all the chat rules and decision-making

# Create the main FastAPI application
app = FastAPI(title="Fight Chaplain Backend")

# --- ★★★ NEW: Startup Event ★★★ ---
@app.on_event("startup")
async def on_startup():
    """
    This function runs *once* when the FastAPI server starts.
    We use it to pre-calculate and load the crisis phrase
    embeddings into memory so they are ready for instant use.
    """
    logging.info("Server is starting up...")
    logic.initialize_crisis_embeddings()
# --- End New Block ---

# Set up CORS (Cross-Origin Resource Sharing)
# This is a security feature in browsers. This code basically
# tells any browser, "It's okay, you're allowed to connect to this server."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any website domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all request types (POST, GET, etc.)
    allow_headers=["*"],  # Allow all request headers
)

@app.get("/diag")
async def diagnostics():
    """
    A "diagnostics" or "health check" page.
    
    You can go to this page in your browser (e.g., http://127.0.0.1:8000/diag)
    to quickly check if the server is running and if it successfully
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
    This is the main endpoint that handles chat messages.
    
    It expects a user's message and their chat history. It then:
    1. Figures out the user's state (faith, escalation level).
    2. Decides if it needs to fetch scripture.
    3. Builds instructions (a "system message") for the AI.
    4. Sends it all to the AI and streams the response back.
    """
    try:
        # Get the JSON data that the browser sent
        data = await request.json()
        
        # Pull out the user's new message and the previous chat history
        msg = data.get("message", "").strip()  # .strip() removes any extra spaces
        history = data.get("history", [])
        
        # If the message is empty, it's a mistake, so we stop.
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        # If the data sent wasn't in the right format (JSON), we stop.
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # Create a "SessionState" object. This is a temporary little box
    # to hold all the info about this *specific* conversation.
    s = logic.SessionState(history=history)
    
    # --- This is where we run our core chatbot logic ---
    
    # 1. Look through the old messages to see if a faith was ever mentioned
    for turn in history:
        if turn.get("role") == "user":
            logic.try_set_faith(turn.get("content", ""), s)
            # If we find a specific faith (not the default), we can stop looking
            if s.faith and s.faith != "bible_nrsv":
                break
    
    # 2. Now, check the *new* message for a faith. If none is found
    #    at all, this function will set the default (bible_nrsv).
    logic.try_set_faith(msg, s)
    
    # 3. ★★★ UPDATED: This now runs the new SEMANTIC crisis check ★★★
    logic.update_session_state(msg, s)
    
    # 4. Create the full conversation to send to the AI
    current_conversation = history + [{"role": "user", "content": msg}]

    # 5. ★★★ UPDATED: This now uses the new HyDE logic ★★★
    retrieval_ctx = logic.get_rag_context(msg, s)
    
    # 6. Build the main "system message" (the instructions for the AI)
    sys_msg = logic.system_message(
        s, 
        quote_allowed=bool(retrieval_ctx), 
        retrieval_ctx=retrieval_ctx
    )
    
    # 7. Combine the instructions and the conversation
    messages_for_llm = [sys_msg] + current_conversation

    async def stream_generator():
        """
        This is a special function that sends the AI's response
        back in small pieces (chunks) as they come in, instead of
        making the user wait for the whole thing.
        """
        final_full_response = ""  # We'll build the full response here
        
        try:
            # Start the streaming request to OpenAI
            stream = rag.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages_for_llm,
                stream=True,  # This tells OpenAI to send the response in chunks
                temperature=0.2  # Makes the AI more focused and less random
            )
            
            # Loop through each chunk of text as we receive it
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    final_full_response += content  # Add this piece to our full text
                    
                    # This "yield" sends this piece of text back to the browser
                    # It's formatted as a "server-sent event"
                    yield f"data: {json.dumps({'text': content})}\n\n"
        
        except Exception as e:
            # If something goes wrong *during* the stream
            error_msg = f"Stream Error: {str(e)}"
            logging.error(error_msg, exc_info=True)  # Log the full error for us
            
            # Send a user-friendly error back to the browser
            yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"
            return  # Stop the function
            
        # --- After the stream is finished ---
        
        # Now that we have the complete response, we check if we
        # need to add a crisis referral or a faith leader offer.
        footer_text = logic.apply_referral_footer(final_full_response, s)
        
        if footer_text:
            # If a footer is needed, send it as one last chunk
            yield f"data: {json.dumps({'text': footer_text})}\n\n"
        
        # Send a final message to tell the browser we're all done
        yield "data: [END]\n\n"

    # This tells FastAPI to run our "stream_generator" function
    # and send the streaming data back to the user.
    return StreamingResponse(stream_generator(), media_type="text/event-stream")