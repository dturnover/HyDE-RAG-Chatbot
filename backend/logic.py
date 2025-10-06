# logic.py
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import config
import rag

# SessionState needs to be defined here or in a separate state.py
@dataclass
class SessionState:
    history: List[Dict[str, str]] = field(default_factory=list)
    faith: Optional[str] = None
    @property
    def _chap(self):
        class MockChap:
            escalate = "none"
            turns = len(self.history) // 2
        return MockChap()

def system_message(s: SessionState, quote_allowed: bool, retrieval_ctx: Optional[str]) -> Dict[str, str]:
    print("[DEBUG] logic.py: Entered system_message")
    # ... function body is the same ...
    return {"role": "system", "content": "..."} # Truncated for brevity

def try_set_faith(msg: str, s: SessionState) -> None:
    print("[DEBUG] logic.py: Entered try_set_faith")
    if s.faith: 
        print(f"[DEBUG] logic.py: Faith already set to '{s.faith}'. Skipping.")
        return
    # ... rest of the function is the same ...

def wants_retrieval(msg: str) -> bool:
    # This is simple enough not to need a debug statement
    m_lower = msg.lower()
    return any(word in m_lower for word in config.ASK_WORDS | config.DISTRESS_KEYWORDS)

def get_rag_context(msg: str, s: SessionState) -> Optional[str]:
    print("[DEBUG] logic.py: Entered get_rag_context")
    should_retrieve = wants_retrieval(msg)
    faith_is_set = bool(s.faith)
    print(f"[DEBUG] logic.py: wants_retrieval={should_retrieve}, faith_is_set={faith_is_set}")
    
    if should_retrieve and faith_is_set:
        print(f"[DEBUG] logic.py: Conditions met. Calling rag.hybrid_search for corpus '{s.faith}'")
        hits = rag.hybrid_search(msg, s.faith)
        if hits:
            print(f"[DEBUG] logic.py: RAG search returned {len(hits)} hits.")
            # ... rest of function is the same ...
        else:
            print("[DEBUG] logic.py: RAG search returned 0 hits.")
    return None # Simplified for brevity

# ... update_session_metrics and apply_referral_footer are the same ...