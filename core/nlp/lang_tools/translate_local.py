from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from core.scripts.config_manager import get_config
from core.scripts.tools import Response


class Translator:
    def __init__(self):
        self.config = get_config()['lang_tools']
        self.ru_en_model = None
        self.en_ru_model = None
        self.en_ru_tokenizer = None
        self.ru_en_tokenizer = None

    def load_en_ru_model(self):
        if self.en_ru_model is not None:
            return Response(-1, 'Model is already loaded')
        try:
            self.en_ru_model = AutoModelForSeq2SeqLM.from_pretrained(self.config['en_ru_translator_path'])
            self.en_ru_tokenizer = AutoTokenizer.from_pretrained(self.config['en_ru_translator_path'], src_lang='en')
        except Exception as e:
            return Response(0, 'An error occurred while loading en-ru translation model', e)
        return Response(1, 'en-ru translation model loaded successfully')

    def load_ru_en_model(self):
        if self.ru_en_model is not None:
            return Response(-1, 'Model is already loaded')
        try:
            self.ru_en_model = AutoModelForSeq2SeqLM.from_pretrained(self.config['ru_en_translator_path'])
            self.ru_en_tokenizer = AutoTokenizer.from_pretrained(self.config['ru_en_translator_path'], src_lang='en')
        except Exception as e:
            return Response(0, 'An error occurred while loading ru-en translation model', e)
        return Response(1, 'ru-en translation model loaded successfully')

    def ru_to_en(self, text):
        if self.ru_en_model is None:
            return Response(-1, 'Model is not loaded')
        try:
            t = self.ru_en_tokenizer(text, return_tensors='pt')
            text = self.ru_en_tokenizer.batch_decode(
                self.ru_en_model.generate(**t),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        except Exception as e:
            return Response(0, 'An error occurred while applying ru-en translation', e)
        return Response(1, 'ru-en translation applied successfully', data=text[0])

    def en_to_ru(self, text):
        if self.en_ru_model is None:
            return Response(-1, 'Model is not loaded')
        try:
            t = self.en_ru_tokenizer(text, return_tensors='pt')
            text = self.en_ru_tokenizer.batch_decode(
                self.en_ru_model.generate(**t),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        except Exception as e:
            return Response(0, 'An error occurred while applying en-ru translation', e)
        return Response(1, 'en-ru translation applied successfully', data=text[0])
