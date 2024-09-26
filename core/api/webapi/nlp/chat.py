from core.scripts.config_manager import get_config
from functools import wraps
from urllib.parse import unquote, quote
from core.scripts.tools import Response, api_request


def is_model_loaded():
    return api_request(get_config().get('server_urls').get('nlp_url') + '/nlp/chat/model_loaded')['is_loaded']


def model_check():
    def is_loaded_deco(f):
        @wraps(f)
        def func(*args, **kwargs):
            if is_model_loaded():
                return f(*args, **kwargs)
            else:
                return Response(-1, 'Model is not loaded')

        return func

    return is_loaded_deco


def load_model():
    return api_request(get_config().get('server_urls').get('nlp_url') + '/nlp/chat/reload_model')


def reset_chat():
    return api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/reset")


# @model_check()
# def unload_model():
#     response = md.unload_model()
#     return response


@model_check()
def send_message(msg, new_tokens=60):
    return unquote(
        api_request(get_config().get('server_urls').get('nlp_url') + '/nlp/chat/generate', {'prompt': quote(msg), 'max_tokens': new_tokens})['generated'])


@model_check()
def generate_no_chat(msg, new_tokens=60):
    return unquote(
        api_request(get_config().get('server_urls').get('nlp_url') + '/nlp/chat/raw_generate', {'prompt': quote(msg), 'max_tokens': new_tokens})['generated'])


@model_check()
def regenerate_last_message():
    return unquote(api_request(get_config().get('server_urls').get('nlp_url') + '/nlp/chat/regenerate')['generated'])


# @model_check()
# def load_chat(chat_name):
#     return chat.store.load_chat(chat_name)


@model_check()
def get_chat():
    return api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/history")['chat']


@model_check()
def is_empty():
    return bool(len(api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/history")['chat'][1:]))


@model_check()
def get_chat_name():
    return api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/history")['chat']


@model_check()
def remove_last_message():
    return api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/remove_last_message")


@model_check()
def get_system_message():
    return unquote(api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/system_message")['system_message'])


@model_check()
def change_system_message(msg):
    return api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/set_system_message", {'text': quote(msg)})


@model_check()
def change_last_message(msg):
    return api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/change_last_message", {'text': quote(msg)})


@model_check()
def change_last_user_message(msg):
    return api_request(get_config().get('server_urls').get('nlp_url') + "/nlp/chat/change_last_user_message", {'text': quote(msg)})
