from json import load, dump
from core.scripts.tools import Response
from llama_cpp import ChatCompletionRequestSystemMessage as SystemMessage


class ChatStore:
    def __init__(self):
        self.chat_name = None
        self.system_message = SystemMessage(role='system', content='')
        self.store: list[dict] = []

    @property
    def last_message(self):
        if len(self.store) == 0:
            return Response(-1, 'Chat store is empty')
        return Response(1, data=self.messages[-1])

    @property
    def messages(self):
        return [self.system_message] + self.store

    def remove_last_message(self):
        if len(self.store) < 1:
            return Response(-1, 'Chat store is empty')
        self.store = self.store[:-1]
        return Response(1)

    def reset_chat(self):
        self.__init__()
        return Response(1, 'Chat reset successfully')

    def change_last_message(self, new_message):
        if len(self.store) == 0:
            return Response(-1, 'Chat store is empty')
        self.store[-1] = new_message
        return Response(1, "Last message changed successfully")

    def save_chat(self, filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                dump({'chat_name': self.chat_name, 'system_message': self.system_message["content"], 'messages': self.store}, f)
        except Exception as e:
            return Response(0, 'An error occurred while saving chat', e)
        return Response(1, "Chat saved successfully")

    def load_from_json(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                chat = load(f)
            self.chat_name = chat['chat_name']
            self.store = chat['messages']
            self.system_message = SystemMessage(role='system', content=chat['system_message'])
        except Exception as e:
            return Response(0, 'And error occurred while loading chat from json', e)
        return Response(1, 'Chat loaded successfully')

    def add_message(self, role: str, message: str):
        if role not in ['user', 'system', 'assistant']:
            return Response(-1, 'Invalid role')
        self.store.append({'role': role, 'content': message})
        return Response(1, 'Message added successfully')

    def add_message_dict(self, message_dict):
        self.store.append(message_dict)
        return Response(1, 'Message added successfully')

    def change_last_user_message(self, new_message):
        if len(self.store) == 0:
            return Response(-1, 'Chat store is empty')
        for index, msg in enumerate(reversed(self.store)):
            if msg['role'] == 'user':
                self.store[-(index+1)]['content'] = new_message
        return Response(1, 'Last user message changed successfully')

    def change_system_message(self, new_message):
        self.system_message['content'] = new_message
        return Response(1, 'System message changed successfully')
