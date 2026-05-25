from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TTSProvider = Literal["openai", "local"]
TTSModel = Literal["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
TTSVoice = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]
TTSResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
TTSPCMFormat = Literal["pcm_s16le"]
TTSPCMChannels = Literal[1, 2]
TTSPCMSampleWidth = Literal[2]
LEGACY_TTS_VOICES = {
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
}
PCM_DEFAULT_VOICE_ALIASES = {"", "default", "marin"}
PCM_DEFAULT_MODEL_ALIASES = {"", "gpt-4o-mini-tts", "tts-1", "tts-1-hd"}


class TextToSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    provider: TTSProvider = "openai"
    model: TTSModel = "gpt-4o-mini-tts"
    voice: TTSVoice = "marin"
    response_format: TTSResponseFormat = "mp3"
    instructions: str | None = Field(default=None, max_length=1200)
    speed: float | None = Field(default=None, ge=0.25, le=4.0)

    @model_validator(mode="after")
    def validate_model_voice(self) -> "TextToSpeechRequest":
        if self.model in {"tts-1", "tts-1-hd"} and self.voice not in LEGACY_TTS_VOICES:
            raise ValueError(f"{self.model} does not support voice {self.voice!r}")
        return self


class TextToSpeechChunk(BaseModel):
    id: str | None = Field(default=None, max_length=80)
    text: str = Field(min_length=1, max_length=4000)


class TextToSpeechPCMRequest(BaseModel):
    chunks: list[TextToSpeechChunk] = Field(min_length=1, max_length=64)
    voice: str = Field(default="default", max_length=80)
    model: str | None = Field(default=None, max_length=4096)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    channels: TTSPCMChannels = 1
    sample_width: TTSPCMSampleWidth = 2
    format: TTSPCMFormat = "pcm_s16le"

    @field_validator("voice", mode="before")
    @classmethod
    def normalize_voice(cls, value: Any) -> str:
        voice = "" if value is None else str(value).strip()
        if voice.lower() in PCM_DEFAULT_VOICE_ALIASES:
            return "default"
        return voice

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: Any) -> str | None:
        model = "" if value is None else str(value).strip()
        if model.lower() in PCM_DEFAULT_MODEL_ALIASES:
            return None
        return model
