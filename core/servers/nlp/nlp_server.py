from typing import Any

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

from core.nlp.chat.LLM import Chat, Model
from core.nlp.lang_tools.enhance import Enhance
from core.nlp.lang_tools.translate_local import Translator
from core.nlp.vectordb.db import MessageDB, VectorDB, Embedder
from urllib.parse import unquote, quote

app = FastAPI()

# Loading model instances
emb = Embedder('E:/Programming/AI_Models/BERT_Models/bert-base-uncased')
vdb = VectorDB('src/long_term_memory_db', emb)
db = MessageDB(vdb)
md = Model()
chat = Chat(md)
e = Enhance()
translator = Translator()
response = e.load_model()
if not response.status:
    raise response.error
response = translator.load_en_ru_model()
if not response.status:
    raise response.error
response = translator.load_ru_en_model()
if not response.status:
    raise response.error
emb.load()

# Pydantic

class Text(BaseModel):
    text: str


class Add_VDB(BaseModel):
    collection_name: str
    vectors: list[list[float]]
    payloads: list[dict[str, Any]]
    uids: list


class Search_VDB(BaseModel):
    collection_name: str
    vectors: list[float]
    top_k: int = 5


class Del_by_id_VDB(BaseModel):
    collection_name: str
    ids: list[str]


class Add_MDB(BaseModel):
    message: str
    role: str


class Search_MDB(BaseModel):
    message: list[float]
    top_k: int = 5


class Del_by_id_MDB(BaseModel):
    message_id: str


class ModelGenerate(BaseModel):
    prompt: str
    max_tokens: int
    temp: float


class ChatRegenerate(BaseModel):
    tokens: int


# Vector Storage

#       Embedder
@app.get('/nlp/embed')
async def embed(message: Text):
    message = message.dict()['text']
    message = unquote(message)
    data = emb(message).squeeze(0).tolist()[0]
    return {'embedding': data}


#       Vector DB
@app.get('/nlp/chat/db/create_collection')
async def create_collection(name: Text):
    name = unquote(name.dict()['text'])
    res = vdb.create_collection(name)
    if res.status == -1:
        return {'status': 0, 'message': quote(res.message)}
    return {'status': 1}


@app.get('/nlp/chat/db/add')
async def add_vdb(body: Add_VDB):
    body = body.dict()
    try:
        vdb.add(
            unquote(body['collection_name']),
            body['vectors'],
            body['payloads'],
            body['uids']
        )
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


@app.get('/nlp/chat/db/search')
async def search_vdb(body: Search_VDB):
    body = body.dict()
    try:
        res = vdb.search(
            unquote(body['collection_name']),
            body['vector'],
            body['top_k']
        )
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1, 'search_result': res}


@app.get('/nlp/chat/db/delete_by_id')
async def del_by_id(body: Del_by_id_VDB):
    body = body.dict()
    try:
        vdb.delete_by_id(
            unquote(body['collection_name']),
            body['ids']
        )
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


@app.get('/nlp/chat/db/delete_collection')
async def del_delete(body: Text):
    try:
        vdb.delete_collection(unquote(body.dict()['text']))
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


#       Message DB

@app.get('/nlp/chat/msg_add')
async def add_mdb(body: Add_VDB):
    body = body.dict()
    try:
        uid = db.add_message(unquote(body['message']), unquote(body['role']))
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1, 'msg_id': uid}


@app.get('/nlp/chat/msg/search')
async def search_mdb(body: Search_MDB):
    body = body.dict()
    try:
        res = db.search(unquote(body['message']), body['top_k'])
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1, 'search_result': res}


@app.get('/nlp/chat/msg/del_by_id')
async def delete_message(body: Del_by_id_MDB):
    body = body.dict()
    try:
        db.delete_message(unquote(body['message_id']))
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


@app.get('/nlp/chat/msg/delete_all')
async def delete_all():
    try:
        db.delete_all_messages()
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


# Translator

@app.get('/nlp/translate/ru_en')
async def user_to_llm(text: Text):
    text = text.dict()['text']
    text = translator.ru_to_en(unquote(text))
    if not text.status:
        return {'status': 0, 'message': text.error}
    return {'status': 1, 'text': quote(text.data)}


@app.get('/nlp/translate/en_ru')
async def llm_to_user(text: Text):
    text = text.dict()['text']
    text = translator.en_to_ru(unquote(text))
    if not text.status:
        return {'status': 0, 'message': text.error}
    return {'status': 1, 'text': quote(text.data)}


# Text Enhance

@app.get('/nlp/enhance')
async def enhance(text: Text):
    text = text.dict()['text']
    text = unquote(text)
    text = e.enhance(text)
    if not text.status:
        return {'status': 0, 'message': text.error}
    return {'status': 1, 'text': quote(text.data)}


# LLM

#   Model
@app.get('/nlp/chat/raw_generate')
async def generate(body: ModelGenerate):
    body = body.dict()
    generated = md.model(unquote(body['prompt']), max_tokens=body['max_tokens'], stop=[f"User:"],
                         temperature=body['temp'], echo=True)
    generated = generated["choices"][0]["text"]
    return {'generated': quote(generated)}

#   Chat

@app.get('/nlp/chat/generate')
async def chat_generate(body: ModelGenerate):
    body = body.dict()
    generate = chat.chat_generate(
        unquote(body['prompt']),
        body['max_tokens']
    )
    return {'generated': quote(generate)}


@app.get('/nlp/chat/regenerate')
async def chat_regenerate(body: ChatRegenerate):
    body = body.dict()
    generate = chat.chat_generate(
        body['max_tokens']
    )
    return {'generated': quote(generate)}


#   Store

@app.get('/nlp/chat/history')
async def get_history():
    return chat.store.chat


@app.get('/nlp/chat/reset')
async def reset():
    chat.store.reset_chat()
    return {'status': 1}


@app.get('/nlp/chat/system_message')
async def get_system_message():
    return {'system_message': quote(chat.store.system_message['content'])}


@app.get('/nlp/chat/set_system_message')
async def set_system_message(body: Text):
    body = body.dict()
    chat.store.change_system_message(body['text'])
    return {'status': 1}


@app.get('/nlp/chat/remove_last_message')
async def remove_last_message():
    res = chat.store.remove_last_message()
    if not res.status:
        return {'status': 0, 'message': quote(res.message)}
    return {'status': 1}


@app.get('/nlp/chat/change_last_message')
async def change_last_message(body: Text):
    body = body.dict()
    res = chat.store.change_last_message(body['text'])
    if res.status == -1:
        return {'status': 0, 'message': quote(res.message)}
    return {'status': 1}


@app.get('/nlp/chat/change_last_user_message')
async def change_last_message(body: Text):
    body = body.dict()
    res = chat.store.change_last_user_message(body['text'])
    if res.status == -1:
        return {'status': 0, 'message': quote(res.message)}
    return {'status': 1}
