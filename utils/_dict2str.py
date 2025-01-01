import json
import base64


def encode_dict(data: dict) -> str:
    json_str = json.dumps(data)
    base64_encoded = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    return base64_encoded


def decode_dict(encoded_str: str) -> dict:
    json_str = base64.b64decode(encoded_str.encode('utf-8')).decode('utf-8')
    return json.loads(json_str)
