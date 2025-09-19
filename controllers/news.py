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

# -------------------- Category-based RSS Feeds --------------------
CATEGORY_RSS_FEEDS = {
    "sports": [
        "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        "https://feeds.bbci.co.uk/sport/rss.xml",
        "https://rss.cnn.com/rss/edition_sport.rss"
    ],
    "news": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/reuters/topNews"
    ],
    "movies": [
        "https://www.hollywoodreporter.com/feed/",
        "https://variety.com/feed/",
        "https://feeds.feedburner.com/TheMovieBlog"
    ],
    "technology": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://rss.cnn.com/rss/edition_technology.rss",
        "https://feeds.reuters.com/reuters/technologyNews"
    ]
}

# -------------------- Gemini LLM Gatekeeper --------------------
GEMINI_API_KEY = 'AIzaSyCpPZ2xRGJGx06tZFw_XfU_RTsBiIF_afg'  # Gemini API key
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

async def gemini_generate_title_summary(news_title: str, news_description: str, category: str):
    """
    Use Gemini to create optimized title and summary for category-specific news
    Returns: (new_title: str, summary: str)
    """
    prompt = f"""
Category: {category}
Original Title: {news_title}
Description: {news_description}

Create a clear, engaging title and a concise 2-sentence summary for this {category} news.
Format:
TITLE: [new title]
SUMMARY: [2-sentence summary]
"""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 150}
            }
            resp = await client.post(
                GEMINI_ENDPOINT,
                headers={"X-Goog-Api-Key": GEMINI_API_KEY},
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts"):
                text = candidates[0]["content"]["parts"][0].get("text", "").strip()

                lines = text.split('\n')
                new_title = news_title
                summary = news_description[:100]

                for line in lines:
                    if line.upper().startswith("TITLE:"):
                        new_title = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("SUMMARY:"):
                        summary = line.split(":", 1)[1].strip()

                return new_title, summary

    except Exception as e:
        logger.error(f"Gemini title/summary generation error: {e}")

    # Fallback
    try:
        summary = summarizer(news_description, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    except:
        summary = news_description[:100]

    return news_title, summary

async def gemini_gatekeeper(news_title: str, news_description: str, user_query: str):
    """
    Ask Gemini: Is this news relevant for the user's alert/query?
    Returns: (is_relevant: bool, summary: str)
    """
    prompt = f"""
User Query: {user_query}
News Title: {news_title}
News Description: {news_description}

Is this news relevant to the user query? Answer YES or NO and provide a 1-sentence summary if YES.
"""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 150}
            }
            resp = await client.post(
                GEMINI_ENDPOINT,
                headers={"X-Goog-Api-Key": GEMINI_API_KEY},
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts"):
                text = candidates[0]["content"]["parts"][0].get("text", "").strip()
                if text.upper().startswith("YES"):
                    summary = text.split(":", 1)[1].strip() if ":" in text else news_title
                    return True, summary
            return False, ""

    except Exception as e:
        logger.error(f"Gemini gatekeeper error: {e}")
        # fallback to Hugging Face summarizer if Gemini fails
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

async def fetch_feeds_by_categories(user_categories: list):
    """
    Fetch RSS feeds only for user-selected categories
    """
    if not user_categories:
        return []

    all_feeds = []
    for category in user_categories:
        if category.lower() in CATEGORY_RSS_FEEDS:
            all_feeds.extend(CATEGORY_RSS_FEEDS[category.lower()])

    if not all_feeds:
        return []

    tasks = [fetch_rss_feed(url) for url in all_feeds]
    results = await asyncio.gather(*tasks)
    categorized_news = []

    for i, feed_result in enumerate(results):
        category = None
        feed_url = all_feeds[i]
        for cat, feeds in CATEGORY_RSS_FEEDS.items():
            if feed_url in feeds:
                category = cat
                break

        for item in feed_result:
            item["category"] = category
            categorized_news.append(item)

    return categorized_news

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

    async def analyze_news(self, news_item, user_query, user_profile, user_categories, alert_data):
        cache_key = generate_cache_key(news_item, user_query)
        cached = await intelligence_cache_collection.find_one({"cache_key": cache_key})
        if cached and datetime.now() - cached.get("created_at", datetime.min) < self.cache_duration:
            return cached["analysis"]

        # Category filtering first - if news doesn't match user categories, skip
        news_category = news_item.get('category', '').lower()
        if news_category not in [cat.lower() for cat in user_categories]:
            return {
                "satisfies_user_query": False,
                "confidence_score": 0.0,
                "entities": [],
                "sentiment": "NEUTRAL",
                "sentiment_score": 0.5,
                "should_notify": False,
                "notification_text": ""
            }

        content = f"{news_item.get('title','')} {news_item.get('description','')}"
        similarity = calculate_similarity(content.lower(), user_query.lower())

        # Enhanced personalization with user alert data
        entities = extract_entities(content)
        boost = 0

        # Check for specific keywords from user's custom questions and followup questions
        followup_keywords = alert_data.get("followup_questions", [])
        for keyword in followup_keywords:
            if keyword.lower() in content.lower():
                boost += 0.3  # Higher boost for specific user interests

        # Check user preferences
        for e in entities:
            if any(p.lower() in e.lower() for p in user_profile.get("preferences", {}).get("favorite_players", [])):
                boost += 0.15
            if any(t.lower() in e.lower() for t in user_profile.get("preferences", {}).get("favorite_teams", [])):
                boost += 0.2
            if any(c.lower() in e.lower() for c in user_profile.get("preferences", {}).get("favorite_channels", [])):
                boost += 0.1

        # Category boost
        if news_category in [cat.lower() for cat in user_categories]:
            boost += 0.15

        # Sub-category specific boost
        sub_categories = alert_data.get("sub_categories", [])
        for sub_cat in sub_categories:
            if sub_cat.lower() in content.lower():
                boost += 0.25

        final_score = min(1.0, similarity + boost)
        sentiment_label, sentiment_score = analyze_sentiment(content)

        # Higher threshold - only highly relevant news goes to Gemini
        if final_score < 0.5:
            return {
                "satisfies_user_query": False,
                "confidence_score": final_score,
                "entities": entities,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "should_notify": False,
                "notification_text": ""
            }

        # Gemini Gatekeeper only for qualified news
        should_notify, gatekeeper_summary = await gemini_gatekeeper(news_item.get("title",""), news_item.get("description",""), user_query)
        if not should_notify:
            final_score = 0.0

        analysis = {
            "satisfies_user_query": similarity > 0.5,
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

    async def process_alerts(self, news_items, alerts, user_profile, user_categories):
        results = []
        for alert in alerts:
            # Create more specific queries from user alert data
            queries = []
            custom_q = alert.get("custom_question", "")
            if custom_q:
                queries.append(custom_q)

            # Add followup questions as additional queries
            followup_qs = alert.get("followup_questions", [])
            for fq in followup_qs:
                queries.append(f"news about {fq}")

            # Fallback query
            if not queries:
                queries = [f"news about {alert.get('main_category','general')}"]

            for news_item in news_items:
                best = None
                for q in queries:
                    # Pass alert data for enhanced personalization
                    analysis = await self.engine.analyze_news(news_item, q, user_profile, user_categories, alert)

                    # Only proceed if highly relevant (score > 0.5)
                    if analysis["should_notify"] and analysis["confidence_score"] > 0.5:
                        # Generate Gemini title and summary only for highly qualified news
                        enhanced_title, enhanced_summary = await gemini_generate_title_summary(
                            news_item.get("title", ""),
                            news_item.get("description", ""),
                            news_item.get("category", "general")
                        )

                        news_result = {
                            **news_item,
                            "alert_id": alert["alert_id"],
                            "user_id": alert["user_id"],
                            "matched_query": q,
                            "ai_analysis": analysis,
                            "final_score": analysis["confidence_score"],
                            "notification_text": analysis["notification_text"],
                            "enhanced_title": enhanced_title,
                            "enhanced_summary": enhanced_summary,
                            "personalization_score": self._calculate_personalization_score(news_item, alert, user_profile)
                        }
                        if not best or news_result["final_score"] > best["final_score"]:
                            best = news_result
                if best:
                    results.append(best)

        # Enhanced deduplication and sorting with personalization
        unique_results, seen_titles = [], set()
        for r in results:
            t = r["title"].lower()
            if t not in seen_titles and r["final_score"] > 0.6:  # Higher threshold
                unique_results.append(r)
                seen_titles.add(t)

        # Sort by combined score (relevance + personalization)
        unique_results.sort(key=lambda x: (x["final_score"] + x.get("personalization_score", 0))/2, reverse=True)
        return unique_results[:15]  # Fewer, more relevant results

    def _calculate_personalization_score(self, news_item, alert, user_profile):
        """Calculate how personalized this news is for the specific user"""
        score = 0
        content = f"{news_item.get('title','')} {news_item.get('description','')}".lower()

        # Check followup questions (user's specific interests)
        followup_keywords = alert.get("followup_questions", [])
        for keyword in followup_keywords:
            if keyword.lower() in content:
                score += 0.4

        # Check custom question relevance
        custom_q = alert.get("custom_question", "")
        if custom_q:
            custom_similarity = calculate_similarity(content, custom_q.lower())
            score += custom_similarity * 0.3

        # Check sub-categories
        sub_cats = alert.get("sub_categories", [])
        for sub_cat in sub_cats:
            if sub_cat.lower() in content:
                score += 0.3

        return min(1.0, score)

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
        # Fetch user alerts to get categories
        alerts = [a async for a in alerts_collection.find({"user_id": user_id, "is_active": True})]
        if not alerts:
            return {"message": "No active alerts", "intelligent_news": []}

        # Extract categories from user's active alerts
        user_categories = set()
        for alert in alerts:
            main_cat = alert.get("main_category", "").lower()
            if main_cat:
                user_categories.add(main_cat)
            sub_cats = alert.get("sub_categories", [])
            for sub_cat in sub_cats:
                if sub_cat:
                    user_categories.add(sub_cat.lower())

        user_categories = list(user_categories)
        if not user_categories:
            return {"message": "No categories found in user alerts", "intelligent_news": []}

        user_profile = await users_collection.find_one({"user_id": user_id}) or {"preferences": {}}

        # Fetch news only for user-selected categories
        category_news = await fetch_feeds_by_categories(user_categories)
        if not category_news:
            return {"message": "No news found for selected categories", "intelligent_news": []}

        # Process only category-filtered news through NLP and Gemini
        intelligent_results = await learning_system.process_alerts(category_news, alerts, user_profile, user_categories)
        await learning_system.create_notifications(intelligent_results, user_id)

        if intelligent_results:
            await personalized_collection.insert_one({
                "user_id": user_id,
                "categories": user_categories,
                "generated_at": datetime.now(),
                "results": intelligent_results
            })

        return {
            "user_id": user_id,
            "categories": user_categories,
            "intelligent_news": intelligent_results
        }

    except Exception as e:
        logger.error(f"Error generating intelligent news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)
