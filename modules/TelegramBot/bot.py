import os
from asyncio import sleep
from threading import Thread

import dotenv
from telebot import TeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from core.scripts.config_manager import get_config
from modules.TelegramBot.decorators import Decorators
from modules.TelegramBot.handler import Handler, MessageStack, Answering, chat
from time import sleep
from random import choice

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
stack = MessageStack()
answering = Answering()

delay = [1, 1.2, 0.75, 1.7, 2, 1.5]
r_symbols = ['\"', '\'', '<', '>']

handler.prepare_chat()

bot.send_message(get_config()['telegram_bot']['user_id'], 'Я онлайн!')


def del_message(message, id, time):
    def del_info_msg(time):
        sleep(time)
        bot.delete_message(message.chat.id, id.id)

    del_info = Thread(target=del_info_msg, args=(time,))
    del_info.start()


def send_message(message, text, time: int = 20):
    id = bot.send_message(message.chat.id, text, parse_mode='html', disable_web_page_preview=True)
    if not time == -1:
        del_message(message, id, time)


def is_answering(f):
    def wrapped(message, *args, **kwargs):
        if not answering.state:
            return f(message, *args, **kwargs)
        send_message(message, '<b>Я отвечаю, будь терпелив</b>', time=5)
    return wrapped


def prepare_chat_history(message: str):
    for i in r_symbols:
        message = message.replace(i, '\\' + i)
    return message


@dec.restricted()
@bot.message_handler(commands=['start'])
def start_cmd(message):
    bot.delete_message(message.chat.id, message.id)
    send_message(message, 'Начнём общение!', -1)


@dec.restricted()
@bot.message_handler(commands=['getchat'])
def get_chat(message):
    bot.delete_message(message.chat.id, message.id)
    msgs = ("".join([f"<b>{i['role'].upper()}</b>: {prepare_chat_history(i['content'])}\n" for i in chat.get_messages()])) if chat.is_empty() else 'Чат пустой!'
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Удалить сообщение", callback_data="delete"))
    bot.send_message(
        message.chat.id,
        f'Имя чата: <b>{chat.get_chat_name()}</b>\n\n'
             f'Промпт: \n\"<i>{chat.current_system_message()["content"]}\"</i>\n\n'
             f'Сообщения: \n\n'
             f'{msgs}',
             reply_markup=markup,
             parse_mode='html'
    )

@is_answering
@dec.restricted()
@bot.message_handler(commands=['regen'])
def regen(message):
    bot.delete_message(message.chat.id, message.id)
    res = stack.pull()
    if not res.status:
        send_message(message, "Я не нашла никаких сообщений!", 5)
        return None
    for i in reversed(res.data['msgs']):
        bot.delete_message(message.chat.id, i.id)

    return text_message_handler(res.data['user_msg'], regenerate=True)


@is_answering
@dec.restricted()
@bot.message_handler(commands=['rem2'])
def remove2(message):
    bot.delete_message(message.chat.id, message.id)
    res = stack.pull()
    if not res.status:
        send_message(message, "Я не нашла никаких сообщений!", 5)
        return None
    for i in reversed(res.data['msgs']):
        bot.delete_message(message.chat.id, i.id)
    bot.delete_message(message.chat.id, res.data['user_msg'].id)

    for i in range(2):
        res = chat.remove_last_message()
        if not res.status:
            return send_message(message, "Я не нашла никаких сообщений!", 5)


@is_answering
@dec.restricted()
@bot.message_handler(content_types=['voice'])
def voice_message_handler(message):
    answering.set_state(True)
    voice_message_info = bot.get_file(message.voice.file_id)
    voice_file = bot.download_file(voice_message_info.file_path)
    temp, transcribed = handler.handle_voice(voice_file)
    voice, text = temp
    bot.reply_to(message, f'Я услышала: "<i>{transcribed}</i>"', parse_mode='html')
    ids = {'msgs': [], 'user_msg': message.id}
    for msg in text:
        ids['msgs'].append(bot.send_message(message.chat.id, msg))
        sleep(choice(delay))
    ids['msgs'].append(bot.send_voice(message.chat.id, voice.getvalue()))
    stack.push(ids)
    answering.set_state(False)


@is_answering
@dec.restricted()
@bot.message_handler(content_types=['text'])
def text_message_handler(message, regenerate=False):
    answering.set_state(True)
    voice, text = handler.handle_text(message.text, regenerate)
    ids = {'msgs': [], 'user_msg': message}
    for msg in text:
        sleep(choice(delay))
        ids['msgs'].append(bot.send_message(message.chat.id, msg))
    ids['msgs'].append(bot.send_voice(message.chat.id, voice.getvalue()))
    stack.push(ids)
    answering.set_state(False)


@bot.callback_query_handler(func=lambda call: call.data == "delete")
def callback_delete_message(call):
    bot.delete_message(call.message.chat.id, call.message.message_id)


def run():
    print('Starting...')
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as E:
            print(E)
            print('Restarting in 5 seconds...')
            sleep(5)
