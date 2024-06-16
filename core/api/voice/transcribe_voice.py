import json
import queue
import wave
from io import BytesIO

import numpy as np
import sounddevice as sd
import time

from scipy.io import wavfile

from core.voice.VoiceProcessor.stt import stt_model
from numpy import ndarray

q = queue.Queue()

def q_callback(indata, frames, time, status):
    q.put(bytes(indata))

def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

def transcribe_realtime(timeout: int=20, debug: bool=False) -> str:
    """
    API function that captures voice from microphone and transcribes it into text

    :param timeout: Time interval when speech recognition will happen in seconds(-1 is unlimited time)
    :param debug: Show debug info
    :return: Transcribed text as string
    """
    with sd.RawInputStream(samplerate=16000, blocksize=8000, device=1,
                           dtype='int16', channels=1, callback=q_callback):
        lts = time.time()
        condition = (lambda: time.time() - lts <= timeout) if timeout > 0 else True
        while condition:
            data = q.get()
            if stt_model.model.AcceptWaveform(data):
                res = json.loads(stt_model.model.Result())['text']
                if res != "":
                    print(f"Text: {res}" if debug else "", end="\n" if debug else "")
                    q.empty()
                    return res

def transcribe_audio(audio: ndarray):
    """
    API function that transcribes audio into text

    :param audio: audio as ndarray
    :param debug: Show debug info
    :return: Transcribed text as string
    """
    audio = float2pcm(audio)
    wav_mem = BytesIO()
    wavfile.write(wav_mem, 16000, audio)
    wav_mem.seek(0)
    wf = wave.open(wav_mem, "rb")
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        stt_model.model.AcceptWaveform(data)

    return json.loads(stt_model.model.FinalResult())['text']
