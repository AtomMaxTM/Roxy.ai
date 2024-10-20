from urllib.parse import quote

import numpy as np
from core.scripts.config_manager import get_config
from core.scripts.tools import api_request


def generate_raw_voice(text: str, ssml: bool = False):
    """
    Voice API function that generates raw voice via silero tts

    :param ssml: ssml
    :param text: text to generate voice
    :return: Generated voice as numpy array
    """
    return np.array(api_request(
        get_config().get('server_urls').get('tts_url') + '/tts_silero',
        {"text": quote(text), "ssml": ssml}
    )['voice'])


def tts(text: str, ssml: bool = False):
    """
    Voice API function that implements tts pipeline

    :param text: text to generate voice from
    :param ssml: enables generation from text with ssml tags
    :return: voice as numpy array
    """

    return np.array(
            api_request(
                get_config().get('server_urls').get('tts_url') + '/tts', {"text": quote(text), "ssml": ssml}
            )['voice']
        )
