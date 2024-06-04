from llama_cpp import Llama
from core.scripts.tools import Response
from core.scripts.config_manager import get_config

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
            self.model = Llama(model_path=self.model_path, n_ctx=self.context_size, channels)
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
        self.message_stack = [{"role": "system", "content": "You are ai assistant named Roxy who communicates via messenger with her creator Max. Make responses not big and split them on smaller messages via <NEWMSG> word."}]
        print(self.llm.load_model())

    def chat_mode_generation(self, text, token_amount=40):
        # generated = self.llm.model(f"Max: {text}.\nRoxy: ", max_tokens=token_amount, stop=[f"Max:"], temperature=0.7, echo=True)
        self.message_stack.append({"role": "user", "content": text})
        generated = self.llm.model.create_chat_completion(self.message_stack, stop=[f"Max:"], temperature=0.7, max_tokens=token_amount)
        generated = generated["choices"][0]["message"]['content']
        self.message_stack.append({"role": "assistant", "content": generated})
        print(self.message_stack)
        return generated

    def raw_mode_generation(self, text, token_amount=40):
        generated = self.llm.model(f"User: {text}.\nAssistant: ", max_tokens=token_amount, stop=[f"User:"], temperature=0.7, echo=True)
        generated = generated["choices"][0]["text"]
        return generated

