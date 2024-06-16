from core.nlp.lang_tools.translate_local import Translator

translator = Translator()
response = translator.load_en_ru_model()
if not response.status:
    raise response.error

response = translator.load_ru_en_model()
if not response.status:
    raise response.error


def user_to_llm(text):
    text = translator.ru_to_en(text)
    if not text.status:
        raise text.error
    return text.data


def llm_to_user(text):
    text = translator.en_to_ru(text)
    if not text.status:
        raise text.error
    return text.data
