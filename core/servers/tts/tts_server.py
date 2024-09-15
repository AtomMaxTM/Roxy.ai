from typing import Any

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

from core.voice.VoiceGen.tts import TTS

app = FastAPI()

tts = TTS()

# Pydantic

class Text(BaseModel):
    text: str
    ssml: bool


# TTS

@app.get('/tts')
async def tts_api(text: Text):
    text = text.dict()
    voice = tts(text['text'], text['ssml'])

    if not voice.status:
        return {'status': 0, 'error': voice.error}
    return {'status': 1, 'voice': voice.data}
