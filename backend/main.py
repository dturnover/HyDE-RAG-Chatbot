# main.py
import json
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import config, rag, logic, state

app = FastAPI(title="Fight Chaplain Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/diag")
async def diagnostics():
    """Endpoint to check backend status and configuration."""
    return {
        "status": "ok",
        "openai_model": config.OPENAI_MODEL,
        "embedding_model": config.EMBED_MODEL,
        "rag_path": str(config.RAG_DATA_PATH),
        "available_corpora": list(rag.AVAILABLE_CORPORA.keys())
    }

@app.post("/chat")
async def chat_handler(request: Request, response: Response):
    """Main chat endpoint, supporting streaming via query parameter."""
    sid, s = state.get_or_create_session(request, response)
    
    try:
        data = await request.json()
        msg = data.get("message", "").strip()
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # --- Main Logic Flow ---
    logic.try_set_faith(msg, s)
    logic.update_session_metrics(msg, s)
    s.history.append({"role": "user", "content": msg})

    # Trim history to manage context window size
    if len(s.history) > 20:
        s.history = s.history[:1] + s.history[-19:]

    retrieval_ctx = logic.get_rag_context(msg, s)
    
    sys_msg = logic.system_message(s, quote_allowed=bool(retrieval_ctx), retrieval_ctx=retrieval_ctx)
    
    messages_for_llm = [sys_msg] + s.history[1:]

    # --- Streaming Response ---
    async def stream_generator():
        full_response = ""
        if not rag.client:
            yield "data: [DEV MODE] OpenAI client not configured.\n\n"
            return
            
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
                    # Use json.dumps to handle special characters correctly in SSE
                    yield f"data: {json.dumps({'text': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
            
        # Append footer and save final response
        final_response = logic.apply_referral_footer(full_response, s._chap.escalate)
        footer = final_response[len(full_response):]
        if footer:
             yield f"data: {json.dumps({'footer': footer})}\n\n"

        s.history.append({"role": "assistant", "content": final_response})
        yield "data: [END]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)