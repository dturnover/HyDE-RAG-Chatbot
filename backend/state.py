# state.py
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from fastapi import Request, Response

# --- Session State Data Structures ---
@dataclass
class _ChaplainState:
    """Internal state for the chaplain's escalation logic."""
    turns: int = 0
    distress_hits: int = 0
    crisis_hits: int = 0
    escalate: str = "none"
    faith_pref: str = ""

@dataclass
class SessionState:
    """Represents the state of a user session."""
    history: List[Dict[str, str]] = field(default_factory=lambda: [{"role": "system", "content": ""}])
    rollup: Optional[str] = None
    faith: Optional[str] = None
    _chap: _ChaplainState = field(default_factory=_ChaplainState)

# Global session storage (simple in-memory dictionary)
SESSIONS: Dict[str, SessionState] = {}

def get_or_create_session(request: Request, response: Response) -> Tuple[str, SessionState]:
    """
    Gets or creates a session ID and its corresponding state.
    Sets a secure, cross-site compatible cookie on the response if the session is new.
    """
    sid = request.headers.get("X-Session-Id") or request.cookies.get("sid")
    is_new = not sid or sid not in SESSIONS

    if is_new:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = SessionState()
        # Set a secure cookie for stateful interactions
        response.set_cookie(
            "sid", sid,
            httponly=True,
            samesite="none",
            secure=True,
            path="/"
        )

    return sid, SESSIONS[sid]