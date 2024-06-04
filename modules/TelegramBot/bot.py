import os
import dotenv
from telebot import TeleBot
from core.scripts.config_manager import get_config
from modules.TelegramBot.decorators import Decorators
from modules.TelegramBot.handler import Handler

# this is temporary solution to secure bot token
# this will be removed on release
# Proper code:
token = get_config()['telegram_bot']['token']
# -----------------------------
dotenv.load_dotenv()
token = os.environ.get('token')
# -----------------------------

bot = TeleBot(token)
dec = Decorators()
handler = Handler()

@dec.restricted()
@bot.message_handler(content_types=['voice'])
def voice_message_handler(message):
    voice_message_info = bot.get_file(message.voice.file_id)
    voice_file = bot.download_file(voice_message_info.file_path)
    voice, text = handler.handle_voice(voice_file)
    bot.send_voice(message.chat.id, voice.getvalue())
    bot.send_message(message.chat.id, text)

@dec.restricted()
@bot.message_handler(content_types=['text'])
def voice_message_handler(message):
    text = handler.handle_text(message.text)
    bot.send_message(message.chat.id, text)

def start():
    print('Starting...')
    bot.polling(non_stop=True)