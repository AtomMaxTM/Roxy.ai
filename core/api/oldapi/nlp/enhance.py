from typing import Literal

from core.nlp.lang_tools.enhance import Enhance

e = Enhance()
response = e.load_model()
if not response.status:
    raise response.error

def enhance(
        text: str,
        lang:Literal['en', 'de', 'ru', 'es'] = 'ru',
        len_limit=150
) -> str:
    """
    NLP API function that enhances text

    It makes letters capital and puts those ['-', ',', '.', '!', '?'] symbols
    Example:
        Before: 'afterwards we were taken to one of the undamaged dormitory buildings'

        After: 'Afterwards, we were taken to one of the undamaged dormitory buildings.'

    Supported languages: 'en', 'de', 'ru', 'es'

    :param text: raw text to enhance
    :return: enhanced text
    """
    x = e.enhance(text, lang, len_limit)
    if not x.status:
        raise x.error
    return x.data
