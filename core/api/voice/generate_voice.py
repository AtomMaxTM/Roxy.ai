from core.voice.VoiceGen.tts import silero, rvc
from numpy import ndarray


def generate_raw_voice(text: str):
    """
    Voice API function that generates raw voice via silero tts

    :param text: text to generate voice
    :return: Generated voice as numpy array
    """
    return silero.raw_generate(text)


def generate_raw_ssml_voice(text: str):
    """
    Voice API function that generates raw voice with ssml tags via silero tts

    :param text: text to generate voice
    :return: Generated voice as numpy array
    """
    return silero.raw_ssml_generate(text)


def change_voice(raw_voice: ndarray):
    """
    Voice API function that changes voice

    :param raw_voice: raw_voice as numpy array
    :return: new changed voice as numpy array
    """
    return rvc(raw_voice)


def tts(text: str, ssml: bool = False):
    """
    Voice API function that implements tts pipeline

    :param text: text to generate voice from
    :param ssml: enables generation from text with ssml tags
    :return: voice as numpy array
    """
    silero_response = generate_raw_voice(text) if not ssml else generate_raw_ssml_voice(text)
    if silero_response.status in [-1, 0]:
        return silero_response
    return change_voice(silero_response.data)
