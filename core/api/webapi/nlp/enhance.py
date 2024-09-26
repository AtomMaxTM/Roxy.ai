from urllib.parse import unquote, quote

from core.scripts.tools import api_request
from core.scripts.config_manager import get_config


def enhance(text: str) -> str:
    """
    NLP API function that enhances text

    Example:
        Before: 'afterwards we were taken to one of the undamaged dormitory buildings'
        After: 'Afterwards, we were taken to one of the undamaged dormitory buildings.'

    :param text: raw text to enhance
    :return: enhanced text
    """
    return unquote(
        api_request(
            get_config().get('server_urls').get('nlp_url')+'/nlp/enhance',
            {'text': quote(text)}
        )['text']
    )