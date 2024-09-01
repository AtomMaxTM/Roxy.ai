from qdrant_client import QdrantClient, models as qdrant_models
from transformers import pipeline
import os
from typing import List, Dict, Any
from uuid import uuid4
from core.scripts.tools import get_config

path = get_config()['vector_db']['db_path']


class Embedder:
    def __init__(self, model_path):
        if os.path.exists(model_path):
            self.pipe = pipeline('feature-extraction', model_path, return_tensors='pt')
            self.size_embed = self.pipe.model.config.hidden_size
        else:
            raise ValueError(f'{model_path} is invalid model path')

    def __call__(self, text):
        return self.pipe(text)

    @property
    def size(self):
        return self.size_embed

class VectorDB:
    def __init__(self, path: str, embedder: Embedder):
        self.client = QdrantClient(path=path)
        self.embed = embedder

    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(size=self.embed.size, distance=qdrant_models.Distance.COSINE),
        )

    def add(self, collection_name: str, vectors: list[list[float]], payloads: list[dict[str, Any]], uids: list):
        points = [
            qdrant_models.PointStruct(
                id=uid,
                vector=vector,
                payload=payload
            )
            for vector, payload, uid in zip(vectors, payloads, uids)
        ]
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def search(self, collection_name: str, vector: list[float], top_k: int = 5):
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k
        )
        return search_result

    def delete_by_id(self, collection_name: str, ids: List[str]):
        self.client.delete(collection_name=collection_name, points_selector=qdrant_models.PointIdsList(points=ids))

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name=collection_name)


class MessageDB:
    def __init__(self, db: VectorDB, collection_name: str = 'messages'):
        self.db = db
        self.collection_name = collection_name

    def add_message(self, message: str, role: str) -> str:
        unique_id = str(uuid4())
        vector = self.db.embed(message)[:, 0, :]
        payload = {"message": message, "role": role, "id": unique_id}
        self.db.add(self.collection_name, [vector], [payload], [unique_id])
        return unique_id

    def search_messages(self, message: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vector = self.db.embed(message)[:, 0, :]
        search_results = self.db.search(self.collection_name, vector, top_k)
        return [{"message": result.payload['message'], "id": result.payload['id']} for result in search_results]

    def delete_message(self, message_id: str):
        self.db.delete_by_id(self.collection_name, [message_id])

    def delete_all_messages(self):
        self.db.delete_collection(self.collection_name)
        self.db.create_collection(self.collection_name)
