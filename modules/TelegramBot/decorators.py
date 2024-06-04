from functools import wraps
from core.scripts.config_manager import get_config
from core.scripts.tools import Response

class Decorators:
    def __init__(self):
        self.config = get_config()
        self.id = int(self.config['telegram_bot']['user_id'])
        self.strict = True

    def restricted(self):
        def deco_restrict(f):
            @wraps(f)
            def f_restrict(message, *args, **kwargs):
                userid = message.from_user.id
                if (userid == self.id) or not self.strict:
                    return f(message, *args, **kwargs)
            return f_restrict
        return deco_restrict

    def update_config(self):
        try:
            self.config = get_config()
            self.id = self.config['telegram_bot']
        except Exception as e:
            return Response(0, 'An error occurred while updating config', e)
        return Response(1, 'Config updated successfully')
