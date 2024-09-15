from typing import Any

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

from core.voice.VoiceProcessor.stt import stt_model

app = FastAPI()

# Pydantic

class Text(BaseModel):
    text: str
    ssml: bool


# STT


