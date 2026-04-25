"""Standalone voice engine settings.

Drop into any pydantic-settings BaseSettings class by inheriting VoiceSettings,
or instantiate it directly for standalone use.

Example — merge into your app settings:

    from voice_engine.config import VoiceSettings

    class AppSettings(VoiceSettings, BaseSettings):
        my_app_field: str = "hello"
        ...
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_stt_model: str = "gpt-4o-mini-transcribe"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "coral"
    openai_tts_speed: float = 1.3
    openai_tts_instructions: str = (
        "Speak naturally and conversationally at a comfortable pace. "
        "Sound warm and engaged, like talking to a friend."
    )

    # --- Deepgram ---
    stt_provider: str = "deepgram"       # "deepgram" | "openai"
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"

    # --- Session ---
    session_jwt_secret: str = "change-me-dev-secret"
    session_ttl_minutes: int = 60
    session_max_messages: int = 4

    # --- Redis (optional — graceful degradation if unavailable) ---
    redis_url: str = "redis://localhost:6379/0"
    redis_session_enabled: bool = True
    partial_stt_enabled: bool = True
    partial_stt_chunk_ttl_sec: int = 30

    # --- WebRTC ---
    enable_webrtc_audio: bool = True
    enable_webrtc_uplink: bool = True

    # --- Audio cache ---
    audio_dir: str = "static/audio"
    tts_retention_minutes: int = 120
    tts_max_files: int = 300
    max_audio_bytes: int = 10_485_760

    # --- Rate limiting ---
    rate_limit_window_sec: int = 60
    rate_limit_voice_per_window: int = 20

    @property
    def audio_path(self) -> Path:
        return Path(self.audio_dir)
