import redis
import json
import os

class RedisService:
    def __init__(self, host=None, port=6379, db=0):
        if host is None:
            host = os.environ.get('REDIS_HOST', 'localhost')
        self.r = redis.Redis(host=host, port=port, db=db)

    def get_session(self, session_id):
        session_data = self.r.get(session_id)
        if session_data:
            return json.loads(session_data)
        return None

    def save_session(self, session_id, session_data, ttl=None):
        self.r.set(session_id, json.dumps(session_data), ex=ttl)

    def delete_session(self, session_id):
        self.r.delete(session_id)

redis_service = RedisService()
