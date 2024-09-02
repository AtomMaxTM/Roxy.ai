from core.nlp.chat.LLM import Chat, Model
from core.scripts.config_manager import get_config
from functools import wraps

from core.scripts.tools import Response

md = Model()
chat = Chat(md)


def model_check():
    def is_loaded_deco(f):
        @wraps(f)
        def func(*args, **kwargs):
            if md.model is not None:
                return f(*args, **kwargs)
            else:
                return Response(-1, 'Model is not loaded')

        return func

    return is_loaded_deco


def load_model(path=get_config()['llm']['model_path']):
    response = md.load_model(path)
    return response


def reset_chat():
    return chat.store.reset_chat()


@model_check()
def unload_model():
    response = md.unload_model()
    return response


@model_check()
def get_chat():
    return chat.store.chat

@model_check()
def get_messages():
    return chat.store.messages

@model_check()
def is_empty():
    return chat.store.is_empty

@model_check()
def get_chat_name():
    return chat.store.chat_name

@model_check()
def remove_last_message():
    return chat.store.remove_last_message()

@model_check()
def current_system_message():
    return chat.store.system_message

@model_check()
def change_system_message(msg):
    return chat.store.change_system_message(msg)


@model_check()
def change_last_message(msg):
    return chat.store.change_last_message(msg)

@model_check()
def change_last_user_message(msg):
    return chat.store.change_last_user_message(msg)


@model_check()
def send_message(msg, new_tokens=60):
    return chat.chat_generate(msg, new_tokens)


@model_check()
def generate_no_chat(msg, new_tokens=60):
    return chat.generate(msg, new_tokens)


@model_check()
def regenerate_last_message():
    return chat.regenerate_last()

@model_check()
def save_chat_to_file(chat_name):
    return chat.store.save_chat(get_config()['chat']['chat_name'] + chat_name)

@model_check()
def save_chat():
    return chat.store.save_chat(get_config()['chat']['chat_name'] + chat.store.chat_name)

@model_check()
def load_chat(chat_name):
    return chat.store.load_chat(chat_name)

@model_check()
def load_system_message(filename):
    return chat.store.load_system_message(filename)

@model_check()
def get_system_message(filename):
    return chat.store.get_system_message_from_file(filename)
