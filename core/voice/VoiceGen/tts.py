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
            return Response(0, 'An error occurred while using so-vits-svc', e, data=None)
        return Response(1, "Voice generated successfully", data=voice)

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
            return Response(-1, 'Model is not loaded', data=None)
        try:
            synt = self.model.apply_tts(
                ssml_text=text,
                speaker=self.config["tts_speaker"],
                sample_rate=int(self.config['tts_sample_rate']),
                put_accent=True,
                put_yo=True
            ).numpy()
        except Exception as e:
            return Response(0, 'An error occurred while generating raw voice with ssml via silero', e, data=None)
        return Response(1, "Raw voice with ssml was generated successfully", data=synt)


    def raw_generate(self, text: str):
        if self.model is None:
            return Response(-1, 'Model is not loaded', data=None)
        try:
            synt = self.model.apply_tts(
                text=text,
                speaker=self.config["tts_speaker"],
                sample_rate=int(self.config['tts_sample_rate']),
                put_accent=True,
                put_yo=True
            ).numpy()
        except Exception as e:
            return Response(0, 'An error occurred while generating raw voice via silero', e, data=None)
        return Response(1, "Raw voice was generated successfully", data=synt)

    def reset(self) -> Response:
        self.config = get_config()
        self.model = None
        return Response(1, 'Silero model was reset successfully')


class XTTS_Model:
    def __init__(self):
        tts_config = get_config()["xtts"]
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self.config = XttsConfig()
        self.config.load_json(tts_config['model_path'] + '/config.json')
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=tts_config['model_path'] + '/', eval=True)
        self.model.cuda()

    def __call__(self, text: str):
        config = get_config()["xtts"]
        try:
            outputs = self.model.synthesize(
                text,
                self.config,
                speaker_wav=config['ref_path'],
                language=config['lang'],
                temperature=float(config.get('temperature', 0.8)),
                top_k=int(config.get('top_k', 50)),
                top_p=float(config.get('top_p', 0.85)),
                gpt_cond_len=int(config.get('gpt_cond_len', 30))
            )
        except Exception as e:
            return Response(0, 'An error occurred while generating voice with xTTSv2', e, data=None)
        return Response(1, "Voice was generated successfully", data=outputs['wav'])


class TTS:
    def __init__(self):
        self.config = get_config()["tts"]
        engine_type = self.config["engine_type"]
        if engine_type in ('silero', 'rvc'):
            self.silero = SileroModel()
            silero_load = self.silero.load_model()
            if not silero_load.status:
                raise silero_load.error
        if engine_type == 'rvc':
            self.rvc = RVC_Model()
        elif engine_type == 'xtts':
            self.xtts = XTTS_Model()


    def __call__(self, text: str, ssml: bool = False):
        if self.config['engine_type'] in ['silero', 'rvc']:
            res = self.silero.raw_generate(text) if not ssml else self.silero.raw_ssml_generate(text)
            if res.status in [-1, 0]:
                return res
            if self.config['engine_type'] == 'silero':
                return res.data
            else:
                return self.rvc(res.data)
        else:
            return self.xtts(text)
