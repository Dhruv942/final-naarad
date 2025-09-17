from fastapi import APIRouter, Query
import httpx
import feedparser
import asyncio
import logging
from sentence_transformers import SentenceTransformer, util
from db.mongo import alerts_collection

router = APIRouter()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentence Transformer Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Feeds by category
category_feeds = {
    "Sports": [
        "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        "https://rss.cnn.com/rss/edition_sport.rss"
    ],
    "News": [
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://feeds.bbci.co.uk/news/rss.xml"
    ],
    "Movies": [
        "https://www.hollywoodreporter.com/t/movies/feed/"
    ]
}

# ---- RSS Fetch with retry ----
async def fetch_rss_feed(feed_url: str, retries=2):
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(feed_url)
                response.raise_for_status()
            feed = feedparser.parse(response.content)
            return [
                {
                    "title": e.get("title", ""),
                    "description": e.get("description", ""),
                    "link": e.get("link", ""),
                    "published": e.get("published", ""),
                    "content": f"{e.get('title','')} {e.get('description','')}"
                }
                for e in feed.entries[:20]
            ]
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed to fetch {feed_url}: {e}")
            if attempt == retries:
                return []

# ---- Semantic match ----
def check_match(content: str, keywords, threshold=0.4):
    if not keywords:
        return None
    content_lower = content.lower()
    for kw in keywords:
        if not kw:
            continue
        kw_lower = kw.lower()
        if kw_lower in content_lower:
            return {"matched": True, "keyword": kw, "similarity": 1.0}
        try:
            sim = util.pytorch_cos_sim(
                model.encode(kw_lower, convert_to_tensor=True),
                model.encode(content_lower, convert_to_tensor=True)
            ).item()
            if sim > threshold:
                return {"matched": True, "keyword": kw, "similarity": sim}
        except Exception as e:
            logger.warning(f"Semantic match failed for keyword {kw}: {e}")
            continue
    return None

# ---- Fetch all feeds concurrently ----
async def fetch_all_feeds(feeds):
    tasks = [fetch_rss_feed(url) for url in feeds]
    results = await asyncio.gather(*tasks)
    # Flatten list of lists
    return [item for sublist in results for item in sublist]

# ---- Main Route ----
@router.get("/personalized-news")
async def get_personalized_news(user_id: str = Query(None, description="Optional user ID to fetch specific alerts")):
    query = {"is_active": True}
    if user_id:
        query["user_id"] = user_id

    alerts_cursor = alerts_collection.find(query)
    alerts = []
    async for alert in alerts_cursor:
        alert["_id"] = str(alert["_id"])
        alerts.append(alert)

    if not alerts:
        return {"news": [], "total_count": 0, "alerts_processed": 0}

    matched_news = []

    # Process alerts in batches to avoid memory spikes
    batch_size = 50
    for i in range(0, len(alerts), batch_size):
        batch = alerts[i:i+batch_size]
        batch_tasks = []
        for alert in batch:
            main_cat = alert.get("main_category")
            feeds = category_feeds.get(main_cat, [])
            if feeds and alert.get("followup_questions"):
                batch_tasks.append(fetch_all_feeds(feeds))
        # Fetch RSS feeds concurrently
        batch_results = await asyncio.gather(*batch_tasks)

        # Match alerts to news
        for alert, news_items_list in zip(batch, batch_results):
            followups = alert.get("followup_questions", [])
            for item in news_items_list:
                match = check_match(item.get("content", ""), followups)
                if match:
                    matched_news.append({
                        **item,
                        "alert_id": alert["_id"],
                        "user_id": alert["user_id"],
                        "category": alert.get("main_category"),
                        "matched_keyword": match["keyword"],
                        "relevance_score": match.get("similarity", 1.0)
                    })

    # Sort by relevance
    matched_news.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return {
        "news": matched_news[:50],
        "total_count": len(matched_news),
        "alerts_processed": len(alerts)
    }
