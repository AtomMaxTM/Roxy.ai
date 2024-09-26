from dataclasses import dataclass
from typing import Any
import logging as log

import sounddevice as sd
from core.scripts.config_manager import get_config
from datetime import datetime
from requests import get

samplerate = int(get_config()['tts']['sts_sample_rate'])

log.basicConfig(filename=get_config()['log']['logfile'], level=log.ERROR,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


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


def say(data, sr=samplerate):
    sd.play(data, samplerate=sr)
    sd.wait()
    sd.stop()

def api_request(address: str, request_body: dict = None) -> dict:
    return get(address, json=request_body).json()

def generate_template(template):
    current_date = datetime.now()
    formatted_date = template.replace("YYYY", current_date.strftime("%Y")) \
                             .replace("YY", current_date.strftime("%Y")[2:]) \
                             .replace("MM", current_date.strftime("%m")) \
                             .replace("DD", current_date.strftime("%d")) \
                             .replace(":", '.')
    return formatted_date


def log_error(r: Response):
    log.error(f"{r.message} | Error: {str(r.error)}")
