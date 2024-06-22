from functools import wraps
from json import load, dump
from core.scripts.config_manager import get_config
from core.scripts.tools import Response, generate_template, log_error
from llama_cpp import ChatCompletionRequestSystemMessage as SystemMessage



def autosave(f):
    def wrapped(self, *args, **kwargs):
        if not self.autosave_on:
            return f(self, *args, **kwargs)
        fn = f(self, *args, **kwargs)
        res = self.save_chat(self.chat_name)
        if not res.status:
            log_error(res)
        return fn
    return wrapped


class ChatStore:
    def __init__(self, autosave_on=True, chat_name=None):
        self.chat_name = chat_name if chat_name is not None else generate_template(get_config()["chat"]["chat_name_template"])
        self.system_message = SystemMessage(role='system', content='')
        self.store: list[dict] = []
        self.autosave_on = autosave_on

    @property
    def last_message(self):
        if len(self.store) == 0:
            return Response(-1, 'Chat store is empty')
        return Response(1, data=self.chat[-1])

    @property
    def is_empty(self):
        return len(self.store)

    @property
    def chat(self):
        return [self.system_message] + self.store

    @property
    def messages(self):
        return self.store

    @autosave
    def remove_last_message(self):
        if len(self.store) < 1:
            return Response(0, 'Chat store is empty')
        self.store = self.store[:-1]
        return Response(1, 'Last message removed successfully')

    @autosave
    def reset_chat(self):
        self.__init__()
        return Response(1, 'Chat reset successfully')

    @autosave
    def change_last_message(self, new_message):
        if len(self.store) == 0:
            return Response(-1, 'Chat store is empty')
        self.store[-1] = new_message
        return Response(1, "Last message changed successfully")

    def save_chat(self, filename: str):
        try:
            with open(get_config()["chat"]["chats"] + filename + '.json', 'w', encoding='utf-8') as f:
                dump({'chat_name': self.chat_name, 'system_message': self.system_message["content"], 'messages': self.store}, f)
        except Exception as e:
            res = Response(0, 'An error occurred while saving chat', e)
            log_error(res)
            return res
        return Response(1, "Chat saved successfully")

    def load_from_json(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                chat = load(f)
            self.chat_name = chat['chat_name']
            self.store = chat['messages']
            self.system_message = SystemMessage(role='system', content=chat['system_message'])
        except Exception as e:
            res = Response(0, 'And error occurred while loading chat from json', e)
            log_error(res)
            return res
        return Response(1, 'Chat loaded successfully')

    @autosave
    def add_message(self, role: str, message: str):
        if role not in ['user', 'system', 'assistant']:
            return Response(-1, 'Invalid role')
        self.store.append({'role': role, 'content': message})
        return Response(1, 'Message added successfully')

    @autosave
    def add_message_dict(self, message_dict):
        self.store.append(message_dict)
        return Response(1, 'Message added successfully')

    @autosave
    def change_last_user_message(self, new_message):
        if len(self.store) == 0:
            return Response(-1, 'Chat store is empty')
        for index, msg in enumerate(reversed(self.store)):
            if msg['role'] == 'user':
                self.store[-(index+1)]['content'] = new_message
        return Response(1, 'Last user message changed successfully')

    @autosave
    def change_system_message(self, new_message):
        self.system_message['content'] = new_message
        return Response(1, 'System message changed successfully')

    @autosave
    def load_system_message(self, filename):
        try:
            with open(get_config()['chat']['prompts'] + filename, 'r', encoding='utf-8') as f:
                system = f.readline()
            self.system_message = SystemMessage(role='system', content=system)
        except Exception as e:
            res = Response(0, 'And error occurred while loading prompt', e)
            log_error(res)
            return res
        return Response(1, 'Chat loaded successfully')

    def get_system_message_from_file(self, filename):
        try:
            with open(get_config()['chat']['prompts'] + filename, 'r', encoding='utf-8') as f:
                system = f.readline()
        except Exception as e:
            res = Response(0, 'And error occurred while loading prompt', e)
            log_error(res)
            return res
        return Response(1, f'Successfully loaded {filename}.txt prompt', data=system)
