from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List
from uuid import uuid4

import jwt
from fastapi import HTTPException

from voice_engine.config import VoiceSettings as Settings


GREETING_TEXT = "Hi! I am Luma. Main Hindi aur English dono samajh sakti hoon. Aap kaise help chahte hain?"


@dataclass
class SessionState:
    session_id: str
    history: List[dict]


class SessionTokenManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.secret = settings.session_jwt_secret
        self.ttl = timedelta(minutes=max(1, settings.session_ttl_minutes))
        self.max_messages = max(2, settings.session_max_messages)

    def start_session(self) -> tuple[str, str, str]:
        session_id = uuid4().hex
        greeting = GREETING_TEXT
        history = [{"role": "assistant", "content": greeting}]
        token = self.encode(session_id=session_id, history=history)
        return session_id, token, greeting

    def encode(self, session_id: str, history: List[dict] | None = None) -> str:
        """Issue a thin JWT containing only session_id and expiry.
        History is stored server-side in Redis; the JWT is purely an auth token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sid": session_id,
            "iat": int(now.timestamp()),
            "exp": int((now + self.ttl).timestamp()),
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def decode(self, token: str) -> SessionState:
        """Verify JWT signature/expiry and return SessionState with empty history.
        History must be loaded separately from Redis by the caller."""
        if not token:
            raise HTTPException(status_code=401, detail="session token is required")
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(status_code=401, detail="session expired") from exc
        except jwt.InvalidTokenError as exc:
            raise HTTPException(status_code=401, detail="invalid session token") from exc

        sid = str(payload.get("sid", "")).strip()
        if not sid:
            raise HTTPException(status_code=401, detail="invalid session token payload")
        # History is loaded from Redis by the caller — return empty list here
        return SessionState(session_id=sid, history=[])

    def decode_with_history(self, token: str, history: List[dict]) -> SessionState:
        """Decode JWT and attach externally-loaded history (e.g. from Redis)."""
        state = self.decode(token)
        normalized = [
            item for item in history
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and item.get("content")
        ]
        state.history = normalized[-self.max_messages:]
        return state
