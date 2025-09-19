from fastapi import FastAPI, APIRouter, Query, HTTPException
from pydantic import BaseModel
import httpx, feedparser, asyncio, logging
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import motor.motor_asyncio
from datetime import datetime, timedelta
import hashlib
import os

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- MongoDB Setup --------------------
MONGO_DETAILS = "mongodb://localhost:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = client.stagin_local
users_collection = db.get_collection("users")
alerts_collection = db.get_collection("alerts")
personalized_collection = db.get_collection("personalized_news")
notifications_collection = db.get_collection("notifications")
intelligence_cache_collection = db.get_collection("intelligence_cache")

# -------------------- Hugging Face Models --------------------
logger.info("Loading Hugging Face models...")
sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# -------------------- FastAPI App --------------------
app = FastAPI()
router = APIRouter()

# -------------------- Gemini LLM Gatekeeper --------------------
GEMINI_API_KEY = 'AIzaSyCpPZ2xRGJGx06tZFw_XfU_RTsBiIF_afg'
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

async def gemini_gatekeeper(news_title: str, news_description: str, user_query: str):
    """
    Ask Gemini: Is this news relevant for the user's alert/query?
    Returns: (is_relevant: bool, summary: str)
    """
    prompt_text = f"""
User Query: {user_query}
News Title: {news_title}
News Description: {news_description}

Is this news relevant to the user query? Answer YES or NO and provide a 1-sentence summary if YES.
"""

    request_body = {
        "instances": [
            {"input_text": prompt_text}
        ],
        "parameters": {
            "max_output_tokens": 150
        }
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                GEMINI_ENDPOINT,
                headers={"X-Goog-Api-Key": GEMINI_API_KEY},
                json=request_body
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Gemini response structure
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", "").strip()
                if text.upper().startswith("YES"):
                    summary = text.split(":", 1)[1].strip() if ":" in text else news_title
                    return True, summary
            return False, ""

    except Exception as e:
        logger.error(f"Gemini gatekeeper error: {e}")
        # fallback to HF summarizer
        try:
            summary = summarizer(news_description, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        except:
            summary = news_description[:60]
        return True, summary


# -------------------- Helper Functions --------------------
async def fetch_rss_feed(feed_url: str, retries=2):
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(feed_url)
                response.raise_for_status()
            feed = feedparser.parse(response.content)
            last_24h = datetime.utcnow() - timedelta(hours=24)
            return [
                {
                    "title": e.get("title",""),
                    "description": e.get("description",""),
                    "link": e.get("link",""),
                    "published": e.get("published",""),
                    "news_id": f"{feed_url}_{hash(e.get('title',''))}"
                } for e in feed.entries
                if 'published_parsed' in e and datetime(*e.published_parsed[:6]) > last_24h
            ][:20]
        except Exception as e:
            logger.warning(f"RSS fetch failed ({feed_url}): {e}")
            if attempt == retries:
                return []

async def fetch_all_feeds(feeds: list):
    tasks = [fetch_rss_feed(url) for url in feeds]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

def calculate_similarity(text1: str, text2: str):
    try:
        emb1 = sem_model.encode(text1)
        emb2 = sem_model.encode(text2)
        return util.pytorch_cos_sim(emb1, emb2).item()
    except:
        return 0.0

def extract_entities(text: str):
    try:
        entities = ner_pipeline(text)
        return [e['entity_group'] for e in entities]
    except:
        return []

def analyze_sentiment(text: str):
    try:
        result = sentiment_pipeline(text)
        return result[0]['label'], result[0]['score']
    except:
        return "NEUTRAL", 0.5

def generate_cache_key(news_item, user_query):
    signature = f"{news_item.get('title','')}{news_item.get('description','')}_{user_query}"
    return hashlib.md5(signature.encode()).hexdigest()

# -------------------- Intelligence Engine --------------------
class IntelligenceEngine:
    def __init__(self):
        self.cache_duration = timedelta(hours=2)

    async def analyze_news(self, news_item, user_query, user_profile):
        cache_key = generate_cache_key(news_item, user_query)
        cached = await intelligence_cache_collection.find_one({"cache_key": cache_key})
        if cached and datetime.now() - cached.get("created_at", datetime.min) < self.cache_duration:
            return cached["analysis"]

        content = f"{news_item.get('title','')} {news_item.get('description','')}"
        similarity = calculate_similarity(content.lower(), user_query.lower())

        # HuggingFace boosts
        entities = extract_entities(content)
        boost = 0
        for e in entities:
            if any(p.lower() in e.lower() for p in user_profile.get("preferences", {}).get("favorite_players", [])):
                boost += 0.1
            if any(t.lower() in e.lower() for t in user_profile.get("preferences", {}).get("favorite_teams", [])):
                boost += 0.15
            if any(c.lower() in e.lower() for c in user_profile.get("preferences", {}).get("favorite_channels", [])):
                boost += 0.1

        final_score = min(1.0, similarity + boost)
        sentiment_label, sentiment_score = analyze_sentiment(content)

        # Gemini Gatekeeper
        should_notify, gatekeeper_summary = await gemini_gatekeeper(news_item.get("title",""), news_item.get("description",""), user_query)
        if not should_notify:
            final_score = 0.0  # override if Gemini says NO

        analysis = {
            "satisfies_user_query": similarity>0.4,
            "confidence_score": final_score,
            "entities": entities,
            "sentiment": sentiment_label,
            "sentiment_score": sentiment_score,
            "should_notify": should_notify,
            "notification_text": gatekeeper_summary
        }

        await intelligence_cache_collection.update_one(
            {"cache_key": cache_key},
            {"$set":{"cache_key": cache_key, "analysis": analysis, "created_at": datetime.now()}},
            upsert=True
        )
        return analysis

# -------------------- Dynamic Learning --------------------
class DynamicLearningSystem:
    def __init__(self):
        self.engine = IntelligenceEngine()

    async def process_alerts(self, news_items, alerts, user_profile):
        results = []
        for alert in alerts:
            queries = [alert.get("custom_question", f"news about {alert.get('main_category','general')}")]
            for news_item in news_items:
                best = None
                for q in queries:
                    analysis = await self.engine.analyze_news(news_item, q, user_profile)
                    if analysis["should_notify"]:
                        news_result = {
                            **news_item,
                            "alert_id": alert["alert_id"],
                            "user_id": alert["user_id"],
                            "matched_query": q,
                            "ai_analysis": analysis,
                            "final_score": analysis["confidence_score"],
                            "notification_text": analysis["notification_text"]
                        }
                        if not best or news_result["final_score"] > best["final_score"]:
                            best = news_result
                if best:
                    results.append(best)

        # Deduplicate & sort
        unique_results, seen_titles = [], set()
        for r in results:
            t = r["title"].lower()
            if t not in seen_titles:
                unique_results.append(r)
                seen_titles.add(t)
        unique_results.sort(key=lambda x: x["final_score"], reverse=True)
        return unique_results[:20]

    async def create_notifications(self, results, user_id):
        notifications = []
        for r in results:
            if r["final_score"]>0.4:
                notifications.append({
                    "user_id": user_id,
                    "alert_id": r["alert_id"],
                    "title": r["notification_text"],
                    "content": {
                        "news_title": r["title"],
                        "news_link": r.get("link",""),
                        "matched_query": r["matched_query"],
                        "entities": r["ai_analysis"]["entities"],
                        "sentiment": r["ai_analysis"]["sentiment"],
                        "confidence_score": r["final_score"]
                    },
                    "priority":"high" if r["final_score"]>0.2 else "medium",
                    "is_read":False,
                    "created_at": datetime.now()
                })
        if notifications:
            await notifications_collection.insert_many(notifications)

# -------------------- FastAPI Routes --------------------
learning_system = DynamicLearningSystem()

@router.get("/fully-dynamic-intelligent-news")
async def get_intelligent_news(user_id: str = Query(...)):
    try:
        alerts = [a async for a in alerts_collection.find({"user_id": user_id, "is_active": True})]
        if not alerts:
            return {"message":"No active alerts", "intelligent_news":[]}
        user_profile = await users_collection.find_one({"user_id": user_id}) or {"preferences":{}}

        all_news = []
        for alert in alerts:
            feeds = alert.get("feeds", ["https://www.espncricinfo.com/rss/content/story/feeds/0.xml"])
            news_items = await fetch_all_feeds(feeds)
            all_news.extend(news_items)

        intelligent_results = await learning_system.process_alerts(all_news, alerts, user_profile)
        await learning_system.create_notifications(intelligent_results, user_id)

        if intelligent_results:
            await personalized_collection.insert_one({
                "user_id": user_id,
                "generated_at": datetime.now(),
                "results": intelligent_results
            })

        return {"user_id": user_id, "intelligent_news": intelligent_results}

    except Exception as e:
        logger.error(f"Error generating intelligent news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)
