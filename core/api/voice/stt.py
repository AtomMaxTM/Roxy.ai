from core.scripts.tools import get_config, api_request
from numpy import ndarray
# import json
# import queue
# import numpy as np
# import sounddevice as sd
# import time
#
# q = queue.Queue()
#
# def q_callback(indata, frames, time, status):
#     q.put(bytes(indata))
#
#
# def transcribe_realtime(timeout: int=20, debug: bool=False) -> str:
#     """
#     API function that captures voice from microphone and transcribes it into text
#
#     :param timeout: Time interval when speech recognition will happen in seconds(-1 is unlimited time)
#     :param debug: Show debug info
#     :return: Transcribed text as string
#     """
#
#     with sd.RawInputStream(samplerate=16000, blocksize=8000, device=1,
#                            dtype='int16', channels=1, callback=q_callback):
#         lts = time.time()
#         condition = (lambda: time.time() - lts <= timeout) if timeout > 0 else True
#         while condition:
#             data = q.get()
#             if stt_model.model.AcceptWaveform(data):
#                 res = json.loads(stt_model.model.Result())['text']
#                 if res != "":
#                     print(f"Text: {res}" if debug else "", end="\n" if debug else "")
#                     q.empty()
#                     return res


def stt(audio: ndarray):
    """
    API function that transcribes audio into text

    :param audio: audio as ndarray
    :return: Transcribed text as string
    """

    return api_request(get_config().get('server_urls').get('stt_url') + '/stt', {'voice': audio.tolist()})['text']
