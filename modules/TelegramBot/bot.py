import os
from asyncio import sleep
from threading import Thread

import dotenv
from icecream import ic
from telebot import TeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from core.scripts.config_manager import get_config
from modules.TelegramBot.tools import Decorators, TempMessage
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


def prepare_chat_history(message: str):
    for i in r_symbols:
        message = message.replace(i, '\\' + i)
    return message


@dec.restricted()
@bot.message_handler(commands=['start'])
def start_cmd(message):
    bot.delete_message(message.chat.id, message.id)
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Удалить сообщение", callback_data="delete"))
    bot.send_message(
        message.chat.id,
        "Начнём общение!",
        reply_markup=markup,
        parse_mode='html'
    )

@dec.restricted()
@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.delete_message(message.chat.id, message.id)
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Удалить сообщение", callback_data="delete"))
    text = """
<b>Привет, я Рокси!</b>

Со мной можно общатся через голосовые и текстовые сообщения.

Список команд:
 ○ /start - Отправляет стартовое сообщение
 ○ /help - Отправляет это сообщение
 ○ /getchat - Показывает текущие параметры чата
 ○ /regen - Генерирует заново последнее сообщение бота
 ○ /rem2 - Удаляет последние сообщения бота и юзера
 ○ /reset - Сбрасывает текущий чат
 ○ /restart - Перезапускает службы бота
    """
    bot.send_message(
        message.chat.id,
        text,
        reply_markup=markup,
        parse_mode='html'
    )


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


@dec.restricted()
@bot.message_handler(commands=['regen'])
def regen(message):
    bot.delete_message(message.chat.id, message.id)
    if answering.state:
        send_message(message, '<b>Я отвечаю, будь терпелив</b>', time=5)
        return
    res = stack.pull()
    if not res.status:
        send_message(message, "Я не нашла никаких сообщений!", 5)
        return None
    for i in reversed(res.data['msgs']):
        bot.delete_message(message.chat.id, i.id)

    ic(res.data)
    print(res.data['user_msg'])
    print(res.data['transcribed'])
    if res.data.get('transcribed') is not None:
        res.data['user_msg'] = (res.data['user_msg'], res.data['transcribed'])
    return text_message_handler(res.data['user_msg'], regenerate=True, regenerate_voice=res.data.get('transcribed') is not None)


@dec.restricted()
@bot.message_handler(commands=['restart'])
def restart(message):
    bot.delete_message(message.chat.id, message.id)
    # if answering.state:
    #     send_message(message, '<b>Я не могу перезапустится пока отвечаю!</b>', time=5)
    #     return
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Нет ❌", callback_data="delete"))
    markup.add(InlineKeyboardButton("Да ✅", callback_data="restart"))
    bot.send_message(
        message.chat.id,
             'Ты уверен что хочешь перезапустить бота?',
             reply_markup=markup,
             parse_mode='html'
    )


@dec.restricted()
@bot.message_handler(commands=['rem2'])
def remove2(message):
    bot.delete_message(message.chat.id, message.id)
    if answering.state:
        send_message(message, '<b>Я отвечаю, будь терпелив</b>', time=5)
        return
    res = stack.pull()
    if not res.status:
        send_message(message, "Я не нашла никаких сообщений!", 5)
        return None
    for i in reversed(res.data['msgs']):
        bot.delete_message(message.chat.id, i.id)
    if res.data.get('transcribed') is not None:
        bot.delete_message(message.chat.id, res.data.get('transcribed').id)
    bot.delete_message(message.chat.id, res.data['user_msg'].id)

    for i in range(2):
        res = chat.remove_last_message()
        if not res.status:
            return send_message(message, "Я не нашла никаких сообщений!", 5)


@dec.restricted()
@bot.message_handler(commands=['reset'])
def reset(message):
    bot.delete_message(message.chat.id, message.id)
    if answering.state:
        send_message(message, '<b>Я не могу удалить чат пока отвечаю</b>', time=5)
        return
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Нет ❌", callback_data="delete"))
    markup.add(InlineKeyboardButton("Да ✅", callback_data="reset"))
    bot.send_message(
        message.chat.id,
             'Ты уверен что хочешь удалить чат?',
             reply_markup=markup,
             parse_mode='html'
    )


@dec.restricted()
@bot.message_handler(content_types=['voice'])
def voice_message_handler(message):
    if answering.state:
        send_message(message, '<b>Я отвечаю, будь терпелив</b>', time=5)
        bot.delete_message(message.chat.id, message.id)
        return
    answering.set_state(True)
    voice_message_info = bot.get_file(message.voice.file_id)
    voice_file = bot.download_file(voice_message_info.file_path)
    temp, transcribed_id = handler.handle_voice(voice_file, bot, message)
    voice, text = temp
    ids = {'msgs': [], 'user_msg': message}
    ids['transcribed'] = transcribed_id
    for msg in text:
        ids['msgs'].append(bot.send_message(message.chat.id, msg))
        sleep(choice(delay))
    ids['msgs'].append(bot.send_voice(message.chat.id, voice.getvalue()))
    stack.push(ids)
    answering.set_state(False)


@dec.restricted()
@bot.message_handler(content_types=['text'])
def text_message_handler(message, regenerate=False, regenerate_voice=False):
    if answering.state:
        send_message(message, '<b>Я отвечаю, будь терпелив</b>', time=5)
        bot.delete_message(message.chat.id, message.id)
        return
    answering.set_state(True)
    ids = {'msgs': [], 'user_msg': message if not regenerate_voice else message[0]}
    if regenerate_voice:
        ids['transcribed'] = message[1]
    voice, text = handler.handle_text(message.text if not regenerate_voice else message[1].text[13:-1], regenerate)
    if regenerate_voice:
        message = message[0]
    for msg in text:
        sleep(choice(delay))
        ids['msgs'].append(bot.send_message(message.chat.id, msg))
    ids['msgs'].append(bot.send_voice(message.chat.id, voice.getvalue()))
    stack.push(ids)
    answering.set_state(False)


@bot.callback_query_handler(func=lambda call: call.data == "delete")
def callback_delete_message(call):
    bot.delete_message(call.message.chat.id, call.message.message_id)


@bot.callback_query_handler(func=lambda call: call.data == 'reset')
def handle_reset(call):
    bot.delete_message(call.message.chat.id, call.message.message_id)
    info = TempMessage(bot, call.message)
    info.create("<b>Удаляю чат...</b>")
    for _ in range(len(stack)):
        res = stack.pull()
        for i in reversed(res.data['msgs']):
            bot.delete_message(call.message.chat.id, i.id)
        if res.data.get('transcribed') is not None:
            bot.delete_message(call.message.chat.id, res.data.get('transcribed').id)
        bot.delete_message(call.message.chat.id, res.data['user_msg'].id)
    chat.reset_chat()
    handler.prepare_chat()
    info.delete()


@bot.callback_query_handler(func=lambda call: call.data == 'restart')
def handle_restart(call):
    bot.delete_message(call.message.chat.id, call.message.message_id)
    answering.set_state(True)
    global handler
    global stack
    info = TempMessage(bot, call.message)
    info.create("<b>Удаляю чат...</b>")
    for _ in range(len(stack)):
        res = stack.pull()
        for i in reversed(res.data['msgs']):
            bot.delete_message(call.message.chat.id, i.id)
        if res.data.get('transcribed') is not None:
            bot.delete_message(call.message.chat.id, res.data.get('transcribed').id)
        bot.delete_message(call.message.chat.id, res.data['user_msg'].id)
    chat.reset_chat()
    info.change('Перезагружаю обработчик...')
    handler = Handler()
    handler.prepare_chat()
    stack = MessageStack()
    info.delete()
    send_message(call.message, 'Система успешно перезагружена!', 5)
    answering.set_state(False)


# Temporary function. It will be refactored later
def run():
    print('Starting...')
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as E:
            print(E)
            print('Restarting in 5 seconds...')
            sleep(5)
