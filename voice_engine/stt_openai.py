import re

from fastapi import HTTPException, UploadFile
from openai import AsyncOpenAI, BadRequestError

from voice_engine.config import VoiceSettings as Settings


class STTService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def _transcribe_content(self, content: bytes, filename: str, lang_hint: str | None = None) -> str:
        if not self.client:
            raise HTTPException(status_code=500, detail="OpenAI API key is not configured for STT")

        if len(content) > self.settings.max_audio_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Audio too large. Max allowed is {self.settings.max_audio_bytes} bytes.",
            )
        if not content or len(content) < 3000:
            raise HTTPException(status_code=400, detail="Audio too short. Please speak for at least 1 second.")

        # Sanitize: only "hi" and "en" are valid hints; anything else falls back to "en".
        _valid_hints = {"hi", "en"}
        effective_hint = lang_hint if lang_hint in _valid_hints else "en"

        # Language strategy:
        # - "hi" hint → language="hi": Whisper outputs Devanagari for Hindi/Hinglish speech ✓
        # - "en" hint → no language lock + strong bilingual prompt: auto-detect biased toward
        #   English/Hindi, avoiding Korean/Japanese auto-detection from background noise.
        if effective_hint == "hi":
            whisper_kwargs: dict = {
                "language": "hi",
                "prompt": (
                    "AI career assistant for Indian job market. "
                    "User speaks in Hindi. "
                    "Transcribe in Devanagari. "
                    "नौकरी सैलरी रेज़्यूमे स्किल रोडमैप इंटर्नशिप कैरियर"
                ),
            }
        else:
            # Auto-detect with English+Hindi vocabulary prompt so Whisper doesn't
            # mistake background noise for Korean, Japanese, or other languages.
            whisper_kwargs = {
                "prompt": (
                    "AI career assistant for Indian job market. "
                    "User speaks in English or Hinglish. "
                    "resume salary LPA skills roadmap internship FAANG startup "
                    "naukri job placement upskill karna chahta hoon"
                ),
            }

        try:
            response = await self.client.audio.transcriptions.create(
                model=self.settings.openai_stt_model,
                file=(filename or "audio.webm", content),
                **whisper_kwargs,
            )
        except BadRequestError as exc:
            raise HTTPException(status_code=400, detail="Invalid audio input.") from exc
        text = getattr(response, "text", "")
        text = text.strip()

        # Hard filter: reject Korean, Japanese, Chinese, or other non-allowed scripts.
        # These indicate background noise mis-detected as a foreign language.
        _non_allowed = re.compile(r"[\u1100-\u11FF\u3040-\u30FF\uAC00-\uD7FF\u4E00-\u9FFF]")
        if _non_allowed.search(text):
            return ""

        # If Hindi speech is transcribed in Perso-Arabic/Urdu script, convert to Devanagari.
        if re.search(r"[\u0600-\u06FF]", text):
            try:
                fix = await self.client.chat.completions.create(
                    model=self.settings.openai_chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Convert input to Devanagari Hindi script only. Keep meaning same. "
                                "Return only converted text."
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                )
                text = (fix.choices[0].message.content or text).strip()
            except Exception:
                pass
        return text

    async def transcribe(self, audio_file: UploadFile, lang_hint: str | None = None) -> str:
        content = await audio_file.read()
        return await self._transcribe_content(
            content=content,
            filename=audio_file.filename or "audio.webm",
            lang_hint=lang_hint,
        )

    async def transcribe_bytes(self, content: bytes, filename: str = "audio.webm", lang_hint: str | None = None) -> str:
        return await self._transcribe_content(content=content, filename=filename, lang_hint=lang_hint)

    async def transcribe_partial(self, content: bytes, filename: str = "partial.webm", lang_hint: str | None = None) -> str:
        """Transcribe a short (~500ms) audio chunk for early context.

        Unlike transcribe_bytes, this never raises — it returns "" on silence,
        errors, or chunks that are too short, so the hot loop is never interrupted.
        """
        if not self.client:
            return ""
        if not content or len(content) < 600:
            return ""
        if len(content) > self.settings.max_audio_bytes:
            return ""
        _valid_hints = {"hi", "en"}
        effective_hint = lang_hint if lang_hint in _valid_hints else "en"
        if effective_hint == "hi":
            whisper_kwargs: dict = {
                "language": "hi",
                "prompt": "AI career assistant for Indian job market. User speaks in Hindi. Transcribe in Devanagari.",
            }
        else:
            whisper_kwargs = {
                "prompt": (
                    "AI career assistant for Indian job market. User speaks in English or Hinglish. "
                    "resume salary LPA skills roadmap internship FAANG startup"
                ),
            }
        try:
            response = await self.client.audio.transcriptions.create(
                model=self.settings.openai_stt_model,
                file=(filename, content),
                **whisper_kwargs,
            )
            text = (getattr(response, "text", "") or "").strip()
            # Reject non-allowed scripts
            _non_allowed = re.compile(r"[\u1100-\u11FF\u3040-\u30FF\uAC00-\uD7FF\u4E00-\u9FFF]")
            if _non_allowed.search(text):
                return ""
            return text
        except Exception:
            return ""
