from typing import Any

from fastapi import FastAPI, APIRouter
from core.nlp.chat.LLM import Chat, Model
from core.scripts.config_manager import get_config
from functools import wraps
from core.nlp.lang_tools.translate_local import Translator
from core.nlp.lang_tools.enhance import Enhance
from core.scripts.tools import Response
from core.nlp.vectordb.db import MessageDB, VectorDB, Embedder
from pydantic import BaseModel


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


app = FastAPI()

vdb_router = APIRouter(
    prefix='/vectordb'
)

# Pydantic

class Embed(BaseModel):
    text: str

class CollectionName(BaseModel):
    name: str

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

class TranslatorMessage(BaseModel):
    message: str


# Vector Storage

#       Embedder
@vdb_router.get('/embed')
async def embed(message: Embed):
    message = message.dict()['text']
    data = emb(message).squeeze(0).tolist()[0]
    return {'embedding': data}


#       Vector DB
@vdb_router.get('/db/create_collection')
async def create_collection(name: CollectionName):
    name = name.dict()['name']
    res = vdb.create_collection(name)
    if res.status == -1:
        return {'status': 0, 'message': res.message}
    return {'status': 1}


@vdb_router.get('/db/add')
async def add_vdb(body: Add_VDB):
    body = body.dict()
    try:
        vdb.add(
            body['collection_name'],
            body['vectors'],
            body['payloads'],
            body['uids']
        )
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


@vdb_router.get('/db/search')
async def search_vdb(body: Search_VDB):
    body = body.dict()
    try:
        res = vdb.search(
            body['collection_name'],
            body['vector'],
            body['top_k']
        )
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1, 'search_result': res}


@vdb_router.get('/db/delete_by_id')
async def del_by_id(body: Del_by_id_VDB):
    body = body.dict()
    try:
        vdb.delete_by_id(
            body['collection_name'],
            body['ids']
        )
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


@vdb_router.get('/db/delete_collection')
async def del_delete(body: CollectionName):
    try:
        vdb.delete_collection(body.dict()['collection'])
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


#       Message DB

@vdb_router.get('/msg/add')
async def add_mdb(body: Add_VDB):
    body = body.dict()
    try:
        uid = db.add_message(body['message'], body['role'])
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1, 'msg_id': uid}


@vdb_router.get('/msg/search')
async def search_mdb(body: Search_MDB):
    body = body.dict()
    try:
        res = db.search(body['message'], body['top_k'])
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1, 'search_result': res}


@vdb_router.get('/msg/del_by_id')
async def delete_message(body: Del_by_id_MDB):
    body = body.dict()
    try:
        db.delete_message(body['message_id'])
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}


@vdb_router.get('/msg/delete_all')
async def delete_all():
    try:
        db.delete_all_messages()
    except Exception as e:
        return {'status': 0, 'message': e}
    return {'status': 1}

# Translator

@app.get('/translate/ru_en')
async def user_to_llm(text: TranslatorMessage):
    text = text.dict()['message']
    text = translator.ru_to_en(text)
    if not text.status:
        return {'status': 0, 'message': text.error}
    return {'status': 1, 'text': text.data}

@app.get('/translate/en_ru')
async def llm_to_user(text: TranslatorMessage):
    text = text.dict()['message']
    text = translator.en_to_ru(text)
    if not text.status:
        return {'status': 0, 'message': text.error}
    return {'status': 1, 'text': text.data}

