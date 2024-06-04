from io import BytesIO

import librosa
from scipy.io import wavfile


class Handler:
    def __init__(self):
        pass

    def handle_text(self, text):
        raise NotImplementedError("Text handling not implemented")

    def handle_voice(self, voice):
        raise NotImplementedError("Voice handling not implemented")
        # raw_voice, _ = librosa.load(BytesIO(voice_file), sr=16000)
        #
        # text = transcribe_audio(voice)
        #
        # voice, resp = change_voice(raw_voice)
        # if not resp.status:
        #     raise resp.error
        # wav_mem = BytesIO()
        # wavfile.write(wav_mem, 44100, voice)
        # wav_mem.seek(0)