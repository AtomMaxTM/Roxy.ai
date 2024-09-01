from llama_cpp import Llama
from core.scripts.tools import Response
from core.scripts.config_manager import get_config
from core.nlp.chat.store import ChatStore



class Model:
    def __init__(self):
        self.st = get_config()["llm"]

        self.model_path = self.st["model_path"]
        self.context_size = int(self.st["context_size"])
        self.temperature = float(self.st["temp"])
        self.model = None

    def load_model(self, path=None):
        if self.model is not None:
            return Response(-1, "Model already loaded")
        try:
            self.model = Llama(model_path=self.model_path if path is None else path, n_ctx=self.context_size, chat_format="pygmalion")
        except Exception as e:
            return Response(0, "Something went wrong while loading the model", e)
        return Response(1, "Model loaded successfully")

    def unload_model(self):
        if self.model is None:
            return Response(-1, "Model isn't loaded")
        try:
            self.model = None
        except Exception as e:
            return Response(0, "Something went wrong while unloading the model", e)
        return Response(1, "Model unloaded successfully")


class Chat:
    def __init__(self, model):
        self.llm = model
        self.store = ChatStore()

    def load_last_chat(self):
        res = self.store.load_from_json(get_config()["chat"]["chats"] + self.store.chat_name + ".json")
        print(res)
        if not res.status:
            res = self.store.save_chat(self.store.chat_name)
        return res

    def chat_generate(self, text, token_amount=60):
        self.store.add_message('user', text)
        generated = self.llm.model.create_chat_completion(
            self.store.chat, stop=[f"User:"], temperature=self.llm.temperature, max_tokens=token_amount
        )
        generated = generated["choices"][0]["message"]['content']
        self.store.add_message('assistant', generated)
        return generated

    def generate(self, text, token_amount=60):
        generated = self.llm.model(f"User: {text}.\nAssistant: ", max_tokens=token_amount, stop=[f"User:"], temperature=self.llm.temperature, echo=True)
        generated = generated["choices"][0]["text"]
        return generated

    def regenerate_last(self, token_amount=60):
        self.store.remove_last_message()
        generated = self.llm.model.create_chat_completion(
            self.store.chat, stop=[f"User:"], temperature=self.llm.temperature, max_tokens=token_amount
        )
        generated = generated["choices"][0]["message"]['content']
        self.store.add_message('assistant', generated)
        return generated
