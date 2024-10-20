from core.scripts.config_manager import get_config
from urllib.parse import quote
from core.scripts.tools import api_request


def add_message(message: str, role: str):
    return api_request(
        get_config().get('server_urls').get('nlp_url') + '/nlp/chat/msg/add',
        {'message': quote(message), 'role': role}
    )


def search(message: str, top_k: int = 5):
    return api_request(
        get_config().get('server_urls').get('nlp_url') + '/nlp/chat/msg/search',
        {'message': message, 'top_k': top_k}
    )


def delete(message_id: str):
    return api_request(
        get_config().get('server_urls').get('nlp_url') + '/nlp/chat/msg/del_by_id',
        {'message_id': message_id}
    )


def delete_all():
    return api_request(
        get_config().get('server_urls').get('nlp_url') + '/nlp/chat/msg/delete_all'
    )
