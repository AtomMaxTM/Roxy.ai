import os

import torch
from core.voice.VoiceGen.so_vits_svc_fork.inference.main import infer
from core.scripts.config_manager import get_config
from core.scripts.tools import Response

class RVC_Model:
    def __init__(self):
        self.config = get_config()['tts']

    def __call__(self, voice):
        try:
            voice = infer(
                input=voice,
                config_path=self.config["sts_config_path"],
                model_path=self.config["sts_model_path"],
                speaker=self.config["sts_speaker"],
                transpose=int(self.config["sts_transpose"]),
                f0_method=self.config["sts_f0_method"]
            )
        except Exception as e:
            return None, Response(0, 'An error occurred while using so-vits-svc', e)
        return voice, Response(1, "Voice generated successfully")

    def reset(self):
        self.config = get_config()

class SileroModel:
    def __init__(self):
        self.config = get_config()['tts']
        self.model = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(6)

    def load_model(self) -> Response:
        if self.model is None:
            try:
                if not os.path.isfile(self.config['tts_model_path']):
                    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                                   self.config['tts_model_path'])
                self.model = torch.package.PackageImporter(self.config['tts_model_path']).load_pickle("tts_models", "model")
                self.model.to(self.device)
            except Exception as e:
                return Response(0, 'Something went wrong while loading model', e)
        else:
            return Response(-1, 'Model is already loaded')
        return Response(1, 'Model was loaded successfully')

    def raw_ssml_generate(self, text: str):
        if self.model is None:
            return None, Response(-1, 'Model is not loaded')
        try:
            synt = self.model.apply_tts(
                ssml_text=text,
                speaker=self.config["tts_speaker"],
                sample_rate=int(self.config['tts_sample_rate']),
                put_accent=True,
                put_yo=True
            ).numpy()
        except Exception as e:
            return None, Response(0, 'An error occurred while generating raw voice with ssml via silero', e)
        return synt, Response(1, "Raw voice with ssml was generated successfully")


    def raw_generate(self, text: str):
        if self.model is None:
            return None, Response(-1, 'Model is not loaded')
        try:
            synt = self.model.apply_tts(
                text=text,
                speaker=self.config["tts_speaker"],
                sample_rate=int(self.config['tts_sample_rate']),
                put_accent=True,
                put_yo=True
            ).numpy()
        except Exception as e:
            return None, Response(0, 'An error occurred while generating raw voice via silero', e)
        return synt, Response(1, "Raw voice was generated successfully")

    def reset(self) -> Response:
        self.config = get_config()
        self.model = None
        return Response(1, 'Silero model was reset successfully')

silero = SileroModel()
rvc = RVC_Model()

silero_load = silero.load_model()
if not silero_load.status:
    raise silero_load.error

class TTS:
    def __init__(self):
        self.config = get_config()["tts"]

    def __call__(self, text: str, ssml: bool = False):
        raw_voice, res = silero.raw_generate(text) if not ssml else silero.raw_ssml_generate(text)
        if res.status in [-1, 0]:
            return None, res
        return rvc(raw_voice)
