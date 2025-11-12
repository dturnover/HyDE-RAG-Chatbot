# main.py
#   This file is the main controller for our chatbot
#   Uses FastAPI to receive chat messages, orchestrates steps and streams final answer


import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

import config
import rag
import logic


# app initialization
app = FastAPI(title="Fight Chaplain Backend")


@app.on_event("startup")
async def on_startup():
    # this function runs once and only once on startup

    logging.info("Server is starting up...")

    # pre-calculate and load the crisis phrase embeddings into memory so they're ready immediately
    logic.initialize_crisis_embeddings()


# security settings for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # !!!!!!*** CHANGE BEFORE DEPLOY ***!!!!
    allow_credentials=True,
    allow_methods=["*"], # allow (POST, GET, etc)
    allow_headers=["*"], # allow all headers
)


@app.get("/diag")
async def diagnostics():
    # health check available at: https://fight-chaplain.onrender.com/diag
    # checks to see if the server is running and connected to OpenAI and Pinecone
    clients_status = {
        "openai_client": "Initialized" if rag.client else "NOT Initialized",
        "pinecone_index": "Initialized" if rag.index else "NOT Initialized",
        "model": config.OPENAI_MODEL
    }
    return {"status": "ok", "clients": clients_status}


@app.post("/chat")
async def chat_handler(request: Request):
    # main endpoint that handles all chat messages
    # most important part of this application

    try:
        data = await request.json()
        msg = data.get("message", "").strip()
        history = data.get("history", [])

        # basic validation
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # --- core chatbot logic ---
    # It was incorrectly inside the 'except' block above.
    # 1. create a new blank "SessionState" object for this specific request
    s = logic.SessionState(history=history)

    # 2. rebuild faith state from history
    # loop through all past messages to find the last faith mentioned
    for turn in history:
        if turn.get("role") == "user":
            logic.try_set_faith(turn.get("content", ""), s)
            # no break statement so the faith is set to the last on mentioned in history

    # 3. now check the new message for a faith update
    logic.try_set_faith(msg, s)

    # 4. run our 2-layer crisis check on the new message
    logic.update_session_state(msg, s)

    # 5. create the full conversation to send to the AI
    current_conversation = history + [{"role": "user", "content": msg}]

    # 6. Get RAG context (if needed)
    #   this runs the HyDE logic then searches Pinecone
    #   retrieval_ctx will be None or a "RETRIEVED PASSAGE: ... " string
    retrieval_ctx = logic.get_rag_context(msg, s)

    # 7. Build the main "system message"
    #   this combines the base prompt, RAG_rules, and the context we just got
    sys_msg = logic.system_message(
        s,
        quote_allowed=bool(retrieval_ctx),
        retrieval_ctx=retrieval_ctx
    )

    # 8. Combine the instructions and the conversation
    messages_for_llm = [sys_msg] + current_conversation

    async def stream_generator():
        # the most complex part, yields each OpenAI response as a chun to the frontend for typewriter effect
        final_full_response = ""

        try:
            # call the OpenAI API in streaming mode
            stream = rag.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages_for_llm,
                stream=True,
                temperature=0.25 # low temperature for more focused answers
            )

            # send each chunk of text as it arrives
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    final_full_response += content
                    # send the chunk back to the frontend
                    yield f"data: {json.dumps({'text': content})}\n\n"

        # It was incorrectly inside the 'for' loop above.
        except Exception as e:
            error_msg = f"Stream Error: {str(e)}"
            logging.error(error_msg, exc_info=True)
            # send an error message to the frontend
            yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"
            return

        # --- after the stream is finished ---

        # 1. Check if we need to add a crisis/referral footer
        footer_text = logic.apply_referral_footer(final_full_response, s)

        # 2. If a footer exists, send it as a new, separate message
        if footer_text:
            yield f"data: {json.dumps({'text': footer_text})}\n\n"

        # 3. Send the end signal to tell the frontend we're done
        yield "data: [END]\n\n"

    # It was incorrectly inside the 'stream_generator' function.
    # return the stream generator to FastAPI, which handles the streaming response
    return StreamingResponse(stream_generator(), media_type="text/event-stream")