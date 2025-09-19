from fastapi import APIRouter
from typing import List, Dict, Any
import logging

router = APIRouter()

# Simple in-memory storage (prod me DB use karna)
stored_news: List[Dict[str, Any]] = []

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 📨 Cron se automatic news receive karega
@router.post("/receive-news")
async def receive_news(payload: Dict[str, Any]):
    logger.info(f"📩 Raw payload: {payload}")
    logger.info(f"📩 Raw payload: {payload}")  # 👈 debug
    news_items = payload.get("news", [])
    if not news_items:
        return {"status": "error", "message": "No news items received"}

    stored_news.extend(news_items)
    logger.info(f"✅ Received {len(news_items)} news items from cron job")

    return {"status": "success", "received_count": len(news_items)}



# 👀 Manually fetch news (GET)
@router.get("/get-news")
async def get_news():
    return {
        "status": "success",
        "total_count": len(stored_news),
        "news": stored_news[-50:]  # last 50 news items
    }
