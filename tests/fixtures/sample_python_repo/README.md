# Sample Python Repo

A minimal app used for testing `codegraph`.

## Modules

- `models.py` — `User`, `Session`, `BaseModel`, `AdminUser`
- `auth.py` — `authenticate()`, `hash_password()`, session management
- `api.py` — `login()`, `logout()`, `register_user()` handlers
- `utils.py` — validation and pagination helpers

## Usage

```python
from sample_python_repo.auth import authenticate
session = authenticate("alice", "password", users)
```

See `authenticate()` for details on the auth flow.
