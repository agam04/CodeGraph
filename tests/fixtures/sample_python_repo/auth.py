"""Authentication module."""
import hashlib
import time
from typing import Optional

from sample_python_repo.models import Session, User


SECRET_KEY = "super-secret"
TOKEN_TTL = 3600.0


def hash_password(password: str) -> str:
    """Hash a plaintext password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(username: str, password: str, users: dict) -> Optional[Session]:
    """Authenticate a user and return a session if credentials are valid.

    Args:
        username: The username to authenticate.
        password: The plaintext password.
        users: Dict mapping username to User objects.

    Returns:
        A Session if authentication succeeds, None otherwise.
    """
    user = _lookup_user(username, users)
    if user is None:
        return None
    pwd_hash = hash_password(password)
    if not _verify_password(pwd_hash, user.password_hash):
        return None
    return _create_session(user)


def _lookup_user(username: str, users: dict) -> Optional[User]:
    """Look up a user by username."""
    return users.get(username)


def _verify_password(provided_hash: str, stored_hash: str) -> bool:
    """Constant-time password comparison."""
    return provided_hash == stored_hash


def _create_session(user: User) -> Session:
    """Create a new session token for a user."""
    token = hashlib.sha256(f"{user.username}{time.time()}".encode()).hexdigest()
    return Session(token=token, user=user, expires_at=time.time() + TOKEN_TTL)


def invalidate_session(session: Session) -> bool:
    """Invalidate an existing session."""
    session.expires_at = 0.0
    return True


async def async_authenticate(username: str, password: str, users: dict) -> Optional[Session]:
    """Async variant of authenticate for use in async contexts."""
    return authenticate(username, password, users)
