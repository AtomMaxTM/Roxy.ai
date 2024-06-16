from dataclasses import dataclass
import sounddevice as sd
from core.scripts.config_manager import get_config
from typing import Any

sr = int(get_config()['tts']['sts_sample_rate'])

@dataclass
class Response:
    status: int
    message: str = None
    error: Exception = None
    data: Any = None

    def __str__(self):
        err = ", Error: " + str(self.error) if self.error is not None else ""
        msg = ", Message: " + self.message if self.message is not None else ""
        data = ", Data: " + self.data if self.data is not None else ""
        return f'Status: {self.status}{msg}{data}{err}'

    def __repr__(self):
        return f'Response(status={self.status}, message="{self.message}", data={self.data}, error={self.error})'

def say(say):
    sd.play(say, samplerate=sr)
    sd.wait()
    sd.stop()