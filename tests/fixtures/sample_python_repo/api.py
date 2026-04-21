"""REST API handlers."""
from typing import Optional

from sample_python_repo.auth import authenticate, invalidate_session
from sample_python_repo.models import User

_USERS: dict = {}
_SESSIONS: dict = {}


def login(username: str, password: str) -> dict:
    """Handle a login request.

    Returns a dict with 'token' on success or 'error' on failure.
    """
    session = authenticate(username, password, _USERS)
    if session is None:
        return {"error": "Invalid credentials"}
    _SESSIONS[session.token] = session
    return {"token": session.token, "username": username}


def logout(token: str) -> dict:
    """Handle a logout request."""
    session = _SESSIONS.pop(token, None)
    if session is None:
        return {"error": "Session not found"}
    invalidate_session(session)
    return {"ok": True}


def get_current_user(token: str) -> Optional[User]:
    """Get the user associated with a session token."""
    session = _SESSIONS.get(token)
    if session is None:
        return None
    return session.user


def register_user(username: str, email: str, password: str) -> dict:
    """Register a new user."""
    from sample_python_repo.auth import hash_password
    if username in _USERS:
        return {"error": "Username already taken"}
    user = User(username=username, email=email, password_hash=hash_password(password))
    _USERS[username] = user
    return {"ok": True, "username": username}
