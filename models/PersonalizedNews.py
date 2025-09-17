from pydantic import BaseModel
from typing import Optional
from bson import ObjectId

class PersonalizedNews(BaseModel):
    user_id: str
    alert_id: str
    feed_link: str
    matched_keyword: str
    relevance_score: float

    class Config:
        json_encoders = {ObjectId: str}
