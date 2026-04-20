"""Utility functions."""
import re
from typing import Any


EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")


def validate_email(email: str) -> bool:
    """Return True if the email looks valid."""
    return bool(EMAIL_REGEX.match(email))


def sanitize_username(username: str) -> str:
    """Strip non-alphanumeric characters from a username."""
    return re.sub(r"[^a-zA-Z0-9_]", "", username)


def paginate(items: list[Any], page: int = 1, page_size: int = 20) -> dict:
    """Return a page of items."""
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": page,
        "total": len(items),
        "pages": max(1, (len(items) + page_size - 1) // page_size),
    }
