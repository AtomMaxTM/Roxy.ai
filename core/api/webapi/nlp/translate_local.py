from core.scripts.tools import api_request
from core.scripts.config_manager import get_config
from urllib.parse import quote, unquote

def user_to_llm(text):
    return unquote(
        api_request(
            get_config().get('server_urls').get('nlp_url')+'/nlp/translate/ru_en',
        {'text': quote(text)}
        )['text']
    )

def llm_to_user(text):
    return unquote(
        api_request(
            get_config().get('server_urls').get('nlp_url')+'/nlp/translate/en_ru',
        {'text': quote(text)}
        )['text']
    )
