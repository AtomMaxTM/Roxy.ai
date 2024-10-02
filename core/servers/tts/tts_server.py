from urllib.parse import unquote

from fastapi import FastAPI
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
    voice = tts(unquote(text['text']), text['ssml'])
    if not voice.status:
        return {'status': 0, 'error': voice.error}
    return {'status': 1, 'voice': voice.data.tolist()}


@app.get('/tts_silero')
async def silero_api(text: Text):
    text = text.dict()

    if not text['ssml']:
        voice = tts.silero.raw_generate(unquote(text['text']))
    else:
        voice = tts.silero.raw_ssml_generate(unquote(text['text']))

    if not voice.status:
        return {'status': 0, 'error': voice.error}
    return {'status': 1, 'voice': voice.data.tolist()}