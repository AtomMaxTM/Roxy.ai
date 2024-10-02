from io import BytesIO
from json import load, dump

import librosa
from scipy.io import wavfile
from core.scripts.config_manager import get_config
from core.api.oldapi.nlp.translate_local import user_to_llm, llm_to_user
from core.api.oldapi.nlp.enhance import enhance
from core.api.oldapi.voice.generate_voice import change_voice, generate_raw_ssml_voice
from core.api.oldapi.voice.transcribe_voice import transcribe_audio
import core.api.oldapi.nlp.chat as chat
from core.scripts.tools import log_error
from icecream import ic

from core.scripts.tools import Response

chat.load_model()


class Answering:
    def __init__(self):
        self.answering = False

    def set_state(self, state: bool):
        self.answering = state

    @property
    def state(self):
        return self.answering


class MessageStack:
    def __init__(self):
        self.messages = []

    def __len__(self):
        return len(self.messages)

    def load_state(self, filename):
        try:
            with open(filename + ".json", 'r', encoding='utf-8') as f:
                self.messages = load(f)['data']
        except Exception as e:
            return Response(0, 'An error occurred while loading message stack', e)
        return Response(1, 'Successfully loaded message stack')

    def save_state(self, filename):
        try:
            with open(filename + ".json", 'w', encoding='utf-8') as f:
                dump({'data': self.messages}, f)
        except Exception as e:
            return Response(0, 'An error occurred while saving the message stack', e)
        return Response(1, 'Successfully saved message stack')

    def reset(self):
        self.messages = []
        return Response(1)

    def push(self, item):
        self.messages.append(item)

    def pull(self):
        if not len(self.messages):
            return Response(0, 'No last messages found', data=None)
        return Response(1, 'Successfully retrieved last message', data=self.messages.pop(-1))


class Handler:
    def __init__(self):
        self.chat = chat.chat

    def handle_text(self, text, regenerate=False):
        ic(text)
        if not regenerate:
            text = " ".join(list(map(user_to_llm, self.preprocess_llm_response(text))))
            ic(text)
            answer = chat.send_message(text)
        else:
            answer = chat.regenerate_last_message()
        ic(answer)
        answer = self.preprocess_llm_response(answer)
        ic(answer)
        answer = list(map(llm_to_user, answer))
        ic(answer)
        messages = [i for j in list(map(self.preprocess_llm_response, answer)) for i in j]
        ic(messages)
        voice = self.gen_voice_message(messages)
        return voice, messages

    def preprocess_llm_response(self, text):
        for i in list(get_config()['chat']['split']):
            text = text.replace(i + " ", i + '<NEWMSG>')
        text = [i.strip() for i in text.split('<NEWMSG>')]
        return text

    def handle_voice(self, voice, bot, message):
        voice, _ = librosa.load(BytesIO(voice), sr=16000)
        text = transcribe_audio(voice)
        text = enhance(text, len_limit=500)
        transcribed_id = bot.reply_to(message, f'Я услышала: "<i>{text}</i>"', parse_mode='html')
        return self.handle_text(text), transcribed_id

    def gen_voice_message(self, messages):
        text = f"<speak><p>{' '.join([f'<s>{i}</s>' for i in messages])}</p></speak>"
        ic(text)
        raw_voice = generate_raw_ssml_voice(
            text
        )
        ic(raw_voice)
        resp = change_voice(raw_voice.data)
        ic(resp)
        if not resp.status:
            raise resp.error
        wav_mem = BytesIO()
        wavfile.write(wav_mem, 44100, resp.data)
        wav_mem.seek(0)
        return wav_mem

    def prepare_chat(self):
        res = self.chat.load_last_chat()
        if not res.status:
            res = self.chat.load_system_message('BasePrompt.txt')
            if not res.status:
                log_error(res)
        if self.chat.store.system_message['content'] == '':
            res = chat.get_system_message('BasePrompt.txt')
            if not res.status:
                log_error(res)
            else:
                chat.change_system_message(res.data)
