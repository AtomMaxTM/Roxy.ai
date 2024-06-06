from llama_cpp import Llama
from core.scripts.tools import Response
from core.scripts.config_manager import get_config
from core.nlu.LLM.chat_store import ChatStore
from llama_cpp.llama_chat_format import format_pygmalion

class LLM:
    def __init__(self, n_gpu_layers=0):
        st = get_config()["llm_settings"]

        self.model_path = st["model_path"]
        self.loaded = False
        self.threads = int(st["threads"])
        self.context_size = int(st["context_size"])
        self.temperature = float(st["temp"])
        self.username = st["username"]
        self.AI_name = st["ai_name"]
        self.n_gpu_layers = n_gpu_layers
        self.model = None

    def load_model(self):
        if not self.loaded:
            try:
                self.model = Llama(model_path=self.model_path, n_ctx=self.context_size)
            except Exception as e:
                print(e)
                return {"status": 0, "msg": "Something went wrong while loading the model"}
            else:
                self.loaded = True
                return {"status": 1, "msg": "Model loaded successfully"}
        else:
            return {"status": -1, "msg": "Model already loaded"}

    def generate(self, text, token_amount=40, mode="text"):
        generated = self.model(f"{self.username}: {text}.\n{self.AI_name}: ", max_tokens=token_amount, stop=[f"{self.username}:"], temperature=self.temperature)
        return self.handle_generated(generated, mode)

    def handle_generated(self, generated, mode):
        if mode == "text":
            text = generated["choices"][0]["text"]
            # with open(f"generated.txt", "a") as f:
            #     print(text.replace("\u200b", ''), file=f)
            return text


class Model:
    def __init__(self):
        st = get_config()["llm_settings"]

        self.model_path = st["model_path"]
        self.threads = int(st["threads"])
        self.context_size = int(st["context_size"])
        self.temperature = float(st["temp"])
        self.username = st["username"]
        self.AI_name = st["ai_name"]
        self.model = None

    def load_model(self):
        if self.model is not None:
            return Response(-1, "Model already loaded")
        try:
            self.model = Llama(model_path=self.model_path, n_ctx=self.context_size, chat_format="pygmalion")
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
        self.chat_store = ChatStore()

    def chat_generate(self, text, token_amount=60):
        self.chat_store.add_message('user', text)
        generated = self.llm.model.create_chat_completion(
            self.chat_store.messages, stop=[f"Max:"], temperature=0.7, max_tokens=token_amount
        )
        generated = generated["choices"][0]["message"]['content']
        self.chat_store.add_message('assistant', generated)
        return generated

    def generate(self, text, token_amount=40):
        generated = self.llm.model(f"User: {text}.\nAssistant: ", max_tokens=token_amount, stop=[f"User:"], temperature=0.7, echo=True)
        generated = generated["choices"][0]["text"]
        return generated

    def temp_set_system_message(self):
        return self.chat_store.change_system_message('You are ai named Roxy who communicates via messenger with her boyfriend Max. Make responses not big.')

    def regenerate_last(self, token_amount=60):
        self.chat_store.remove_last_message()
        generated = self.llm.model.create_chat_completion(
            self.chat_store.messages, stop=[f"Max:"], temperature=0.7, max_tokens=token_amount
        )
        generated = generated["choices"][0]["message"]['content']
        self.chat_store.add_message('assistant', generated)
        return generated