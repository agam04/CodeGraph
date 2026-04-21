"""Data models for the sample app."""
from dataclasses import dataclass


@dataclass
class User:
    """Represents an application user."""

    username: str
    email: str
    password_hash: str
    is_active: bool = True
    role: str = "user"


@dataclass
class Session:
    """An authenticated user session."""

    token: str
    user: User
    expires_at: float


class BaseModel:
    """Base class for all models."""

    def to_dict(self) -> dict:
        """Serialize model to dictionary."""
        return vars(self)

    def validate(self) -> bool:
        """Validate model fields."""
        return True


class AdminUser(User):
    """A user with admin privileges."""

    def __init__(self, username: str, email: str, password_hash: str) -> None:
        super().__init__(username, email, password_hash, is_active=True, role="admin")

    def can_delete(self, resource: str) -> bool:
        """Check if admin can delete a resource."""
        return self.is_active
