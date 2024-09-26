import wave
from json import loads
from typing import Any

import numpy as np
from pydantic import BaseModel
import os
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy.io import wavfile

from core.voice.VoiceProcessor.stt import stt_model

app = FastAPI()

def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

# Pydantic

class Voice(BaseModel):
    voice: list

# STT

@app.get('/stt')
async def tts_api(voice: Voice):
    try:
        voice = voice.dict()['voice']

        audio = float2pcm(np.array(voice))
        wav_mem = io.BytesIO()
        wavfile.write(wav_mem, 16000, audio)
        wav_mem.seek(0)
        wf = wave.open(wav_mem, "rb")
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            stt_model.model.AcceptWaveform(data)

        text = loads(stt_model.model.FinalResult())['text']
    except Exception as e:
        return {'status': 0, 'error': e}

    return {'status': 1, 'text': text}

