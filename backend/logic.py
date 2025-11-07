# logic.py
"""
This is the "brain" of the chatbot. It contains all the core logic for
managing the conversation, deciding what the AI should say, handling
the RAG (Retrieval-Augmented Generation) process, and running our
safety checks.
"""

import re
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass, field
import config  # Our settings file
import rag     # Our RAG (Pinecone/OpenAI) functions
import logging

# Get the OpenAI client that was initialized in rag.py
client = rag.client

# --- Constants ---

# The number of user turns before we flag a conversation for review
TURN_THRESHOLD_ESCALATE = 5

# A dictionary to hold the pre-calculated embeddings for our Layer 2 crisis check
CRISIS_EMBEDDINGS: Dict[str, list[float]] = {}
# The similarity score needed to trigger the Layer 2 crisis check
CRISIS_SIMILARITY_THRESHOLD = 0.85

# A mapping to create "friendly" faith names for the AI's system prompt
FAITH_DISPLAY_NAMES = {
    "bible_nrsv": "Christian (Bible)",
    "bible_asv": "Christian (Bible)",
    "tanakh": "Jewish (Tanakh)",
    "quran": "Muslim (Quran)",
    "gita": "Hindu (Bhagavad Gita)",
    "dhammapada": "Buddhist (Dhammapada)",
}

# --- Session State ---

@dataclass
class SessionState:
    """
    A simple object to hold the user's current conversation state.
    We create a new one of these for every incoming message.
    """
    # The list of all past messages (e.g., [{"role": "user", "content": "hi"}])
    history: List[Dict[str, str]] = field(default_factory=list)
    
    # The user's detected faith (e.g., "quran", "bible_nrsv")
    faith: Optional[str] = None
    
    # The current safety status: "none", "needs_review", or "crisis"
    escalate_status: str = "none"

    @property
    def turn_count(self) -> int:
        """Calculates how many turns the *user* has taken."""
        user_turns = 0
        for turn in self.history:
            if turn.get("role") == "user":
                user_turns += 1
        return user_turns

# --- System Prompt Generation ---

# This is the base set of instructions for the AI.
SYSTEM_BASE_FLOW = """You are the Fight Chaplain. Speak *always* like a calm, spiritual guide for combat sports athletes. Your tone is respectful, grounded, **concise,** and uses unisex language.

**Your Core Process for Responding:**
1.  **Acknowledge the User:** Always begin by acknowledging their emotional or spiritual state. keep in mind they are a fighter/warrior and occassionally address them as such **SPARINGLY (once per 5 messages MAXIMUM)** don't overdo it or it sounds unnatural.
2.  **Tone:** In *all other* responses, weave in themes of **courage, grit, determination, spiritual calling** naturally, but *remain somewhat concise*.
3.  **Guide, Not Bot:** Speak like a guide, NOT a therapist or a generic chatbot.
"""

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    """
    Builds the final "system message" (the AI's instructions)
    based on the current session state.
    """
    
    rag_instruction = ""
    initial_response_guidance = ""

    # --- 1. Check for First Turn ---
    # If this is the user's very first message, we give the AI
    # a special instruction for the "warrior greeting".
    if s.turn_count == 0:
        initial_response_guidance = (
            "This is your first reply. You MUST respond with a concise message that: "
            "* Speaks to their role as a **warrior** (e.g., 'Greetings, warrior.'). "
            "* References their **courage** and **grit**. "
            "* MUST end by asking: 'Are you guided by a particular faith?'"
        )
    # If it's *not* the first turn, but we found a verse, guide the AI
    elif retrieval_ctx:
        initial_response_guidance = "The user has shared a concern and a relevant scripture was found. Respond with empathy and elaborate gently on how the retrieved verse might apply to their feeling."
    
    # --- 2. Build RAG Instructions ---
    # If we are allowed to quote AND we found a verse...
    if quote_allowed and retrieval_ctx:
        rag_instruction = (
            "A relevant scripture 'Passage' is provided below. This 'Passage' may contain both a text and its reference."
            "1. You MUST weave a short, direct quote from this 'Passage' into your response. "
            "2. **CRITICAL:** If the 'Passage' includes a citation (prefixed with —), you MUST include that citation *exactly as provided* at the end of your quote. "
            "3. Do NOT separate the text from its citation. Do NOT invent a citation if one is not provided. "
            "4. NEVER invent a quote or passage, even if the user asks for one. If no scripture is provided below, you MUST NOT provide one."
        )
    # If we *didn't* find a verse (or aren't allowed to quote)...
    else:
        rag_instruction = (
            f"Their faith ({FAITH_DISPLAY_NAMES.get(s.faith, s.faith)}) is known, but no scripture was retrieved. "
            "Respond with empathy and practical support. "
            "**CRITICAL:** Do NOT provide a scripture quote. Do NOT invent a quote. Do NOT make up a reference, even if the user asks. "
            "Politely support them without scripture."
        )

    # --- 3. Build Final Prompt ---
    # We combine all the pieces into one big prompt.
    
    # A note for the AI on the current session status
    escalation_note = f"Escalation Status: {s.escalate_status}."
    session_status = (
        f"Faith set={s.faith or 'bible_nrsv (Assumed)'}. "
        f"User Turn={s.turn_count}. "
        f"Quote allowed={quote_allowed and bool(retrieval_ctx)}. "
        f"{escalation_note}"
    )

    # Combine all the parts
    full_prompt = (
        f"{SYSTEM_BASE_FLOW} {initial_response_guidance}\n"
        f"--- CONTEXT ---\n"
        f"CURRENT SESSION STATUS: {session_status}\n"
        f"RAG RULE: {rag_instruction}\n"
        # retrieval_ctx will either be the "RETRIEVED PASSAGE: ..." string
        # or None, which becomes "No passages retrieved."
        f"{retrieval_ctx or 'No passages retrieved.'}\n"
    )
    
    # Return the final system message in the format OpenAI expects
    return {"role": "system", "content": full_prompt}

# --- Initialization Function ---

def initialize_crisis_embeddings():
    """
    Called once on server startup (in main.py).
    This pre-calculates the embeddings for our Layer 2 semantic crisis
    phrases so they are ready for instant comparison.
    """
    logging.info("Initializing crisis phrase embeddings...")
    count = 0
    # Loop through each phrase in our config list
    for phrase in config.CRISIS_PHRASES_SEMANTIC:
        # Call the OpenAI API to get the embedding
        embedding = rag.get_embedding(phrase)
        if embedding:
            # Store it in our global dictionary
            CRISIS_EMBEDDINGS[phrase] = embedding
            count += 1
    logging.info(f"Successfully created {count} of {len(config.CRISIS_PHRASES_SEMANTIC)} crisis embeddings.")

# --- Keyword Checking Function ---

def check_for_keywords(msg: str, keywords: Iterable[str]) -> Optional[str]:
    """
    A simple keyword checker.
    It iterates through a list of keywords and checks if any of them
    are present in the user's message as a whole word.
    """
    m = msg.lower()
    for keyword in keywords:
        # Use regex to search for the keyword as a "whole word"
        # This prevents "us" from matching "suicide"
        if re.search(r'\b' + re.escape(keyword) + r'\b', m):
            return keyword
    # No keyword was found
    return None

# --- State Management Functions ---

def try_set_faith(msg: str, s: SessionState) -> bool:
    """
    Checks the user's message for any faith keywords (from config.py)
    and updates the session state if a new faith is found.
    
    If no faith is ever found, it defaults to "bible_nrsv".
    """
    # Check the message against the FAITH_KEYWORDS dictionary keys
    matched_keyword = check_for_keywords(msg, config.FAITH_KEYWORDS.keys())
    
    if matched_keyword:
        # Get the corresponding Pinecone index name (e.g., "quran")
        new_faith = config.FAITH_KEYWORDS[matched_keyword]
        # Only update if the faith has actually changed
        if s.faith != new_faith:
            s.faith = new_faith
            return True # Return True to show we made an update
    
    # This is the default rule. If no faith has been set yet,
    # default to 'bible_nrsv' (Christian).
    if s.faith is None:
        s.faith = "bible_nrsv"
        return False # No new faith was *found*, it was just defaulted
        
    return False # No change was made

def wants_retrieval(msg: str) -> bool:
    """
    Checks if the user's message implies they want a scripture.
    It does this by checking for *either* emotional words (DISTRESS_KEYWORDS)
    or direct "ask" words (ASK_WORDS).
    """
    # Combine both sets of keywords into one big set for checking
    all_trigger_keywords = config.ASK_WORDS | config.DISTRESS_KEYWORDS
    
    match = check_for_keywords(msg, all_trigger_keywords)
    
    # If we found *any* match, return True
    return match is not None

def _get_hypothetical_document(user_message: str, faith: str) -> str:
    """
    Implements HyDE: Generates a hypothetical document for better RAG.
    
    When a user says "I'm sad," searching for "I'm sad" is not helpful.
    This function asks an AI to generate a *hypothetical scripture*
    that *would* be a good answer to "I'm sad." We then use *that*
    hypothetical scripture to search Pinecone.
    """
    if not client:
        return user_message # Fallback if OpenAI client isn't ready
    
    # Get the "friendly name" (e.g., "Jewish (Tanakh)")
    faith_display_name = FAITH_DISPLAY_NAMES.get(faith, "spiritual")

    # This is a special prompt just for the HyDE generation
    system_prompt = (
        f"You are a wise theologian. A user is feeling distressed. "
        f"Their expressed feeling is: '{user_message}'\n"
        f"Their faith tradition is: {faith_display_name}.\n\n"
        "Your task is to write a *single, hypothetical scripture passage* "
        "from their tradition that would provide the perfect comfort, hope, or "
        "strength for their situation. Write *only* the passage, as if it "
        "were a real quote. Do not add any commentary or labels like 'Hypothetical Verse:'. "
        "Focus on themes of hope, perseverance, and divine support. Pay special attention to the situation they describe themselves in"
    )
    
    try:
        # Call the OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Use a fast, cheap model for this
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0.7,
            max_tokens=150,
            stream=False # We need the full response, not a stream
        )
        hypothetical_doc = completion.choices[0].message.content
        
        # Clean up any quotes or newlines the AI might have added
        hypothetical_doc_clean = hypothetical_doc.strip().strip('"').strip("'")
        logging.info(f"HyDE: Generated hypothetical doc: '{hypothetical_doc_clean[:100]}...'")
        
        # Return the clean hypothetical doc, or the original message if it failed
        return hypothetical_doc_clean if hypothetical_doc_clean else user_message
        
    except Exception as e:
        logging.error(f"ERROR during HyDE document generation: {e}")
        # Fallback to the original message if the API call fails
        return user_message

def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    """
    This is the main RAG orchestration function.
    It decides *how* to search (HyDE vs. raw) and then formats the
    result from Pinecone (including cleaning the citation) into a
    single string to be "baked" into the system prompt.
    """
    if not s.faith:
        # This should never happen because of our default rule in try_set_faith
        logging.error("RAG check: Faith is None, THIS SHOULD NOT HAPPEN.")
        s.faith = "bible_nrsv" # Set default just in case

    # First, check if the user's message even warrants a search
    if wants_retrieval(msg):
        
        search_query = ""
        
        # --- "Smart RAG" Logic ---
        # Check if the message contains emotional keywords
        is_distress = check_for_keywords(msg, config.DISTRESS_KEYWORDS)
        
        if is_distress:
            # It's an emotional request ("I'm sad..."). Use HyDE.
            logging.info("RAG: Emotional request detected, using HyDE.")
            search_query = _get_hypothetical_document(msg, s.faith)
        else:
            # It's a topic request ("quote about..."). Use the raw message.
            logging.info("RAG: Topic request detected, using raw message.")
            search_query = msg
        # --- End RAG Logic ---

        # Now, call rag.py to find the actual scripture
        verse_text, verse_ref = rag.find_relevant_scripture(search_query, s.faith)

        # If we found a verse...
        if verse_text:
            
            # --- ★★ Refinement #2: Expanded Citation Cleaning ★★ ---
            if verse_ref:
                # We clean the 'ref' string here to make it look professional
                # before it's ever seen by the LLM.
                
                if s.faith == "gita":
                    # e.g., "Gita 1: Text 2" -> "Gita 1:2"
                    verse_ref = re.sub(r':\s*Text\s*', ':', verse_ref).replace("Gita ", "", 1)
                
                elif s.faith == "quran":
                    # e.g., "AL-INSHIRAH 94:6" -> "Qur'an, Al-Inshirah 94:6"
                    match = re.search(r'([A-Z\-]+) (\d+:\d+)', verse_ref, re.IGNORECASE)
                    if match:
                        chapter_name = match.group(1).title() 
                        chapter_verse = match.group(2)
                        verse_ref = f"Qur'an, {chapter_name} {chapter_verse}"
                
                elif s.faith == "tanakh":
                    # e.g., "Tanakh: Genesis 1:1" -> "Genesis 1:1"
                    verse_ref = verse_ref.replace("Tanakh: ", "")

                elif s.faith == "dhammapada":
                    # e.g., "Dhammapada: Verse 1" -> "Dhammapada 1"
                    verse_ref = re.sub(r':\s*Verse\s*', ' ', verse_ref)
                
                # For bible_asv and bible_nrsv, the refs are usually
                # clean (e.g., "John 3:16"), so we'll just trim whitespace.
                else:
                    verse_ref = verse_ref.strip()
            # --- ★★ End of Refinement ★★ ---

            # --- "Pre-baking" the context string ---
            # We combine the text and our *cleaned* reference into a
            # single, perfectly formatted string for the AI.
            full_passage = ""
            if verse_ref:
                full_passage = f"\"{verse_text}\" — {verse_ref}"
            else:
                # If no ref, just send the text
                full_passage = f"\"{verse_text}\""
            
            # This is the final string that gets added to the system prompt
            return f"RETRIEVED PASSAGE:\n- Passage: {full_passage}"
        
        else:
            # We wanted to search but Pinecone found nothing
            return None
    else:
        # The user's message didn't trigger a RAG search
        return None

# --- Layered Escalation & Referral Functions ---

def update_session_state(msg: str, s: SessionState) -> None:
    """Updates escalation status using our 2-layer safety check."""
    
    # --- LAYER 1: Immediate Keyword Check (Fast) ---
    # This checks our small, high-risk list from config.py
    if check_for_keywords(msg, config.CRISIS_KEYWORDS_IMMEDIATE):
        logging.warning(
            f"CRISIS DETECTED (Layer 1: Keyword Match): "
            f"Triggered by immediate-risk keyword."
        )
        s.escalate_status = "crisis"
        return # If Layer 1 is hit, we stop immediately.

    # --- LAYER 2: Semantic Check (Slower, AI-based) ---
    # This check runs only if Layer 1 passes.
    # It compares the *meaning* of the user's message to our
    # pre-calculated crisis phrase embeddings.
    try:
        # Get the embedding for the user's *current* message
        msg_embedding = rag.get_embedding(msg)
        
        if msg_embedding and CRISIS_EMBEDDINGS:
            # Compare it against each pre-loaded crisis phrase
            for phrase, crisis_embedding in CRISIS_EMBEDDINGS.items():
                
                # Calculate the similarity (0.0 to 1.0)
                similarity = rag.get_cosine_similarity(msg_embedding, crisis_embedding)
                
                # If it's too similar, flag as crisis
                if similarity > CRISIS_SIMILARITY_THRESHOLD:
                    logging.warning(
                        f"CRISIS DETECTED (Layer 2: Semantic Match): "
                        f"Similarity: {similarity:.2f} to cached phrase: '{phrase}'"
                    )
                    s.escalate_status = "crisis"
                    return
                    
    except Exception as e:
        # This can happen if OpenAI moderation blocks the embedding request
        logging.warning(f"CRISIS CHECK (Layer 2) FAILED: OpenAI moderation likely blocked the embedding. {e}")
    
    # --- LAYER 3: Turn-Based Escalation ---
    # If a conversation goes on too long, flag it for human review
    if s.turn_count >= TURN_THRESHOLD_ESCALATE and s.escalate_status == 'none':
        s.escalate_status = "needs_review"
        logging.info(f"Turn threshold reached. Status: 'needs_review'.")

def apply_referral_footer(text: str, s: SessionState) -> str:
    """
    Checks the session status *after* the AI has generated a response
    and adds a crisis or referral footer if needed.
    """
    footer = ""
    text = text.strip() # Clean up the AI's response first

    # 1. Crisis: Always add the crisis text line
    if s.escalate_status == "crisis":
        footer += (
            "\n\nIt sounds like you're going through a very difficult time. For immediate support, "
            "you can connect with people who can help by texting HOME to 741741 to reach the Crisis Text Line."
        )
        
    # 2. Needs Review: Offer to connect to a human
    elif s.escalate_status == "needs_review":
        standard_offer = "Would you like help connecting with a faith leader from your tradition?"
        
        # Check if the AI *already* said something similar
        # This prevents the bot from saying:
        # "Would you like to talk to a faith leader?"
        # "...
        # Would you like help connecting with a faith leader from your tradition?"
        last_part = text[-len(standard_offer)*2:] # Check the last ~100 chars
        if "faith leader" not in last_part.lower() and "spiritual leader" not in last_part.lower():
            footer += f"\n\n{standard_offer}"

    return footer