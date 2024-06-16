from core.nlp.lang_tools.translate_google import translator

def user_to_llm(text):
    return translator.translate(text, 'ru', 'en').text

def llm_to_user(text):
    return translator.translate(text, 'ru', 'en').text



