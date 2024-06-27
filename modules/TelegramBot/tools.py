from functools import wraps
from core.scripts.config_manager import get_config
from core.scripts.tools import Response, log_error


class Decorators:
    def __init__(self):
        self.config = get_config()
        self.id = int(self.config['telegram_bot']['user_id'])
        self.strict = True

    def restricted(self):
        def deco_restrict(f):
            @wraps(f)
            def f_restrict(message, *args, **kwargs):
                if type(message) == tuple:
                    return f(message, *args, **kwargs)
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


class TempMessage:
    def __init__(self, bot, message):
        self.bot = bot
        self.message = message
        self.msg = None
        self.__is_created = False

    def create(self, text, reply=None):
        if self.__is_created:
            return Response(-1, 'Message is already created')
        try:
            if reply is not None:
                self.msg = self.bot.reply_to(self.message, text, parse_mode='html')
            else:
                self.msg = self.bot.send_message(self.message.chat.id, text, parse_mode='html')
        except Exception as e:
            res = Response(0, 'An error occured while creating temp message', e)
            log_error(res)
            return res
        self.__is_created = True
        return Response(1, 'Successfully created temp message')

    def delete(self):
        if not self.__is_created:
            return Response(-1, 'Message is not created')
        self.bot.delete_message(self.message.chat.id, self.msg.id)
        self.bot = None
        self.message = None
        self.__is_created = False
        return Response(1, 'Successfully deleted temp message')

    def change(self, text):
        if not self.__is_created:
            return Response(-1, 'Message is not created')
        try:
            self.bot.edit_message_text(
                chat_id=self.message.chat.id,
                message_id=self.msg.id,
                text=text,
                parse_mode='html'
            )
        except Exception as e:
            res = Response(0, 'An error occured while editing temp message', e)
            log_error(res)
            return res
        return Response(1, 'Successfully edited temp message')
