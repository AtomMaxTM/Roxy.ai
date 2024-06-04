from dataclasses import dataclass
from llama_cpp import ChatCompletionRequestMessage as Message
from llama_cpp import ChatCompletionRequestSystemMessage as SystemMessage
from llama_cpp import ChatCompletionRequestAssistantMessage as AssistantMessage
from llama_cpp import ChatCompletionRequestUserMessage as UserMessage

class ChatStore:
    def __init__(self):
        self.store: list[Message] = []

    def get_last_message(self, as_dict=False):
        return self.store[-1] if not as_dict else {''}

    def get_messages(self, as_dict=False):
        return self.store



class Memory:
    def __init__(self):
        self.chat_store = ChatStore()