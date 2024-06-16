from torch import package
from core.scripts.config_manager import get_config
from core.scripts.tools import Response


class Enhance:
    def __init__(self):
        self.model_path = get_config()['lang_tools']['enh_path']
        self.model = None

    def load_model(self):
        if self.model is not None:
            return Response(-1, 'Model is already loaded')
        try:
            imp = package.PackageImporter(self.model_path)
            self.model = imp.load_pickle("te_model", "model")
        except Exception as e:
            return Response(0, 'An error occurred while loading enhancement model', e)
        return Response(1, 'Enhancement model loaded successfully')

    def enhance(self, text, lang='ru', len_limit=150):
        try:
            x = self.model.enhance_text(text, lang, len_limit=len_limit)
        except Exception as e:
            return Response(0, 'An error occurred while applying enhancement', e)
        return Response(1, 'Enhancement applied successfully', data=x)
