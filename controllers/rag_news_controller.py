"""
RAG-Based News Controller
Replaces traditional ML models with Retrieval-Augmented Generation for personalized news intelligence
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging
import asyncio
import json
import re
from datetime import datetime, timezone
import feedparser
import httpx
import pytz

from core.rag_system import RAGSystem
from db.mongo import db  # Use existing MongoDB connection

# Setup logging
logger = logging.getLogger(__name__)

# API Configuration
GEMINI_API_KEY = 'AIzaSyB4vE8BAkg0J0XZ2bvMR9U4iNs3DfeONS0'

# WATI WhatsApp Configuration
WATI_ACCESS_TOKEN = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJiZGVmNjQ0OS02NDU3LTRiNDYtOTM4Mi03YjNiYmRmMmY2NGIiLCJ1bmlxdWVfbmFtZSI6ImFjdHVhbGx5dXNlZnVsZXh0ZW5zaW9uc0BnbWFpbC5jb20iLCJuYW1laWQiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiZW1haWwiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMjgvMjAyNSAxMjowNzo1OCIsInRlbmFudF9pZCI6IjQ1ODkxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.WPoEwLq2UdUs8Rl61SklQMFQ699mj1CqQ2v7iPZunuU'
WATI_BASE_URL = 'https://live-mt-server.wati.io/458913'
WATI_TEMPLATE_NAME = 'sports'
WATI_BROADCAST_NAME = 'sports_290920250931'

# Initialize RAG System
rag_system = RAGSystem(db, GEMINI_API_KEY)

# FastAPI Router
router = APIRouter()

# =============================================================================
# 30-MINUTE CRON JOB FOR AUTOMATED NEWS ALERTS WITH USER PREFERENCES
# =============================================================================

class NewsAlertScheduler:
    """30-minute scheduler for automated news alerts"""

    def __init__(self):
        self.running = False

    async def start_scheduler(self):
        """Start the 30-minute scheduler"""
        self.running = True
        logger.info("🔔 News Alert Scheduler started (30-minute interval)")

        while self.running:
            try:
                await self.process_automated_alerts()
                await asyncio.sleep(1800)  # 30 minutes
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                await asyncio.sleep(1800)  # 30 minutes

    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("🔔 News Alert Scheduler stopped")

    async def check_user_preferences(self, user_id: str) -> dict:
        """Check user notification preferences"""
        try:
            notifications_collection = db.get_collection("notifications")
            user_prefs = await notifications_collection.find_one({"_id": user_id})

            if user_prefs and "notification_preference" in user_prefs:
                return user_prefs["notification_preference"]

            return {"type": "real_time", "custom_time": None}
        except Exception as e:
            logger.error(f"Error getting preferences for {user_id}: {e}")
            return {"type": "real_time", "custom_time": None}

    async def should_send_now(self, preferences: dict) -> bool:
        """Check if notification should be sent now"""
        try:
            notification_type = preferences.get("type", "real_time")

            if notification_type == "real_time":
                return True

            # Get current IST time
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)
            hour, minute = now.hour, now.minute

            if notification_type == "morning_digest":
                return hour == 7 and minute <= 5
            elif notification_type == "evening_digest":
                return hour == 19 and minute <= 5
            elif notification_type == "custom_time":
                custom_time = preferences.get("custom_time")
                if custom_time:
                    try:
                        custom_hour, custom_minute = map(int, custom_time.split(":"))
                        return hour == custom_hour and abs(minute - custom_minute) <= 5
                    except:
                        return False

            return False
        except Exception as e:
            logger.error(f"Error checking timing: {e}")
            return False

    async def process_automated_alerts(self):
        """Process all active alerts with preference checks"""
        try:
            logger.info("⏰ Processing automated alerts...")

            # Get all active alerts
            alerts_collection = db.get_collection("alerts")
            active_alerts = await alerts_collection.find({"is_active": True}).to_list(None)

            if not active_alerts:
                logger.info("No active alerts found")
                return

            # Process unique users
            processed_users = set()
            total_notifications = 0

            for alert in active_alerts:
                user_id = alert.get("user_id")

                if user_id in processed_users:
                    continue

                try:
                    # Check user preferences
                    preferences = await self.check_user_preferences(user_id)

                    if await self.should_send_now(preferences):
                        logger.info(f"📱 Sending alerts to {user_id} ({preferences.get('type')})")

                        # Call existing get_user_news function
                        result = await get_user_news(user_id)

                        if result.get("status") == "success":
                            notifications = len(result.get("whatsapp_notifications", []))
                            total_notifications += notifications
                            logger.info(f"✅ Sent {notifications} notifications to {user_id}")
                        else:
                            logger.info(f"ℹ️ No notifications for {user_id}")

                    processed_users.add(user_id)
                    await asyncio.sleep(1)  # Rate limiting

                except Exception as e:
                    logger.error(f"Error processing {user_id}: {e}")

            logger.info(f"🎯 Processed {len(processed_users)} users, sent {total_notifications} notifications")

        except Exception as e:
            logger.error(f"Error in automated alerts: {e}")

# Global scheduler instance
news_scheduler = NewsAlertScheduler()

async def start_news_scheduler():
    """Start the news scheduler"""
    await news_scheduler.start_scheduler()

def stop_news_scheduler():
    """Stop the news scheduler"""
    news_scheduler.stop_scheduler()

# RSS Feeds Configuration
CATEGORY_RSS_FEEDS = {
    "sports": [
        "w",
        "https://feeds.bbci.co.uk/sport/rss.xml",
        "https://rss.cnn.com/rss/edition_sport.rss"
    ],
    "news": [
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/reuters/topNews"
    ],
    "movies": [
        "https://www.thehindu.com/entertainment/movies/feeder/default.rss",
        "https://variety.com/feed/",
        "https://feeds.feedburner.com/TheMovieBlog"
    ],
    "technology": [
        "https://feeds.feedburner.com/TechCrunch",
        "https://rss.cnn.com/rss/edition_technology.rss",
        "https://feeds.reuters.com/reuters/technologyNews"
    ]
}

# No Pydantic models needed for simple GET endpoint

# Helper Functions
async def fetch_rss_feed(url: str) -> List[Dict[str, Any]]:
    """Fetch and parse RSS feed"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                articles = []

                for entry in feed.entries[:20]:  # Limit to 20 articles per feed
                    # Extract image from RSS feed
                    image_url = ''

                    # Check multiple possible image sources in RSS
                    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
                        image_url = entry.media_thumbnail[0]['url']
                    elif hasattr(entry, 'media_content') and entry.media_content:
                        image_url = entry.media_content[0]['url']
                    elif hasattr(entry, 'enclosures') and entry.enclosures:
                        for enclosure in entry.enclosures:
                            if enclosure.type and 'image' in enclosure.type:
                                image_url = enclosure.href
                                break
                    elif hasattr(entry, 'links'):
                        for link in entry.links:
                            if link.get('type', '').startswith('image/'):
                                image_url = link.href
                                break

                    # Extract from description/summary if image tag exists
                    if not image_url:
                        import re
                        description_text = getattr(entry, 'description', '') + getattr(entry, 'summary', '')
                        img_match = re.search(r'<img[^>]+src="([^"]+)"', description_text)
                        if img_match:
                            image_url = img_match.group(1)

                    article = {
                        'title': getattr(entry, 'title', ''),
                        'url': getattr(entry, 'link', ''),
                        'summary': getattr(entry, 'summary', ''),
                        'description': getattr(entry, 'description', ''),
                        'published_date': getattr(entry, 'published', ''),
                        'author': getattr(entry, 'author', ''),
                        'source': feed.feed.get('title', 'Unknown'),
                        'tags': [tag.term for tag in getattr(entry, 'tags', [])],
                        'image_url': image_url,
                    }
                    articles.append(article)

                return articles
    except Exception as e:
        logger.error(f"Error fetching RSS feed {url}: {e}")

    return []

async def fetch_all_news():
    """Fetch news from all configured RSS feeds"""
    all_articles = []

    for category, feeds in CATEGORY_RSS_FEEDS.items():
        category_articles = []

        # Fetch feeds concurrently
        tasks = [fetch_rss_feed(feed_url) for feed_url in feeds]
        feed_results = await asyncio.gather(*tasks, return_exceptions=True)

        for articles in feed_results:
            if isinstance(articles, list):
                for article in articles:
                    article['category'] = category
                    category_articles.append(article)

        all_articles.extend(category_articles)
        logger.info(f"Fetched {len(category_articles)} articles for category: {category}")

    return all_articles

# API Endpoints

@router.get("/news/{user_id}")
async def get_user_news(user_id: str):
    """Get intelligent personalized news for a user based on their alerts"""
    try:
        logger.info(f"Getting personalized news for user: {user_id}")

        # Step 0: Ensure we have news data in vector store
        await refresh_news_if_needed()

        # Step 1: Get user's alerts to understand their interests
        user_alerts = await db.get_collection("alerts").find({"user_id": user_id}).to_list(None)

        if not user_alerts:
            return {
                "status": "no_alerts",
                "message": "No alerts found for user. Please create alerts first.",
                "user_id": user_id,
                "articles": []
            }

        # Step 2: Process each alert separately and return individual results
        alert_results = []

        for alert in user_alerts:
            if not alert.get("is_active", True):
                continue

            alert_id = alert.get("_id")
            category = alert.get("main_category", "").lower()
            keywords = alert.get("sub_categories", [])
            followup_questions = alert.get("followup_questions", [])
            custom_question = alert.get("custom_question", "")

            logger.info(f"Processing alert {alert_id}: category={category}, keywords={keywords}")

            # Step 3: Build intelligent contextual query from alert data
            alert_query = await build_contextual_query(alert, keywords, followup_questions, custom_question, category)

            # Step 4: Get category-specific articles without cross-contamination
            relevant_articles = await get_alert_specific_articles(alert, alert_query, category)

            # Step 5: Apply intelligent contextual filtering per alert
            alert_context = {
                'keywords': keywords,
                'query': alert_query,
                'category': category,
                'followup_questions': followup_questions,
                'custom_question': custom_question
            }

            # First apply intelligent content filtering
            contextually_relevant = await apply_intelligent_content_filtering(relevant_articles, alert_context)

            # Apply spam filtering for this alert
            contextually_relevant = remove_obvious_spam_per_alert(contextually_relevant, alert)

            # Apply intelligent filtering per alert (max 1 article - BEST ONE ONLY)
            if contextually_relevant:
                contextually_relevant = await final_gemini_perfect_filter(contextually_relevant, [alert], 1)

            # Enhance with alert context
            alert_articles = []
            for article in contextually_relevant:
                # Add alert-specific metadata
                article['alert_id'] = str(alert_id)
                article['alert_query'] = alert_query
                article['alert_info'] = {
                    "alert_id": str(alert_id),
                    "category": category,
                    "keywords": keywords,
                    "relevance_score": article.get('relevance_score', 0)
                }
                alert_articles.append(article)

            # Create individual alert result
            alert_satisfaction = await debug_user_satisfaction(alert_articles, [alert])

            alert_result = {
                "alert_id": str(alert_id),
                "alert_category": category,
                "alert_keywords": keywords,
                "alert_query": alert_query,
                "total_articles": len(alert_articles),
                "articles": alert_articles,
                "satisfaction_debug": alert_satisfaction
            }

            alert_results.append(alert_result)

        # Step 6: Learn from user alert patterns and update profile
        all_articles_for_learning = []
        for alert_result in alert_results:
            all_articles_for_learning.extend(alert_result['articles'])

        await update_user_profile_from_alerts(user_id, user_alerts, all_articles_for_learning)

        # Step 7: Send WhatsApp notifications via WATI - Send all articles as separate notifications
        # Check for duplicate articles before sending
        sent_notifications_collection = db.get_collection("sent_notifications")
        whatsapp_results = []

        for alert_result in alert_results:
            if alert_result['articles']:  # Only send if articles found
                # Send each article as a separate notification
                for article in alert_result['articles']:
                    article_url = article.get('url', '')

                    # Check if this article was already sent to this user
                    if article_url:
                        existing_notification = await sent_notifications_collection.find_one({
                            "user_id": user_id,
                            "article_url": article_url
                        })

                        if existing_notification:
                            logger.info(f"Skipping duplicate article for {user_id}: {article.get('title', '')[:50]}")
                            whatsapp_results.append({
                                "status": "skipped",
                                "message": "Duplicate article already sent",
                                "article_title": article.get('title', '')
                            })
                            continue

                    # Create individual article result for notification
                    individual_article_result = {
                        **alert_result,
                        'articles': [article]  # Send only one article per notification
                    }
                    wati_response = await send_wati_notification(user_id, individual_article_result)
                    whatsapp_results.append(wati_response)

                    # If successfully sent, record it to prevent future duplicates
                    if wati_response.get("status") == "success" and article_url:
                        await sent_notifications_collection.insert_one({
                            "user_id": user_id,
                            "article_url": article_url,
                            "article_title": article.get('title', ''),
                            "alert_id": alert_result.get('alert_id'),
                            "sent_at": datetime.now(timezone.utc)
                        })

                    # Add small delay between notifications to avoid rate limiting
                    await asyncio.sleep(0.5)

        return {
            "status": "success",
            "user_id": user_id,
            "alerts_processed": len(user_alerts),
            "alert_results": alert_results,
            "total_alerts_returned": len(alert_results),
            "whatsapp_notifications": whatsapp_results,
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Error getting user news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def apply_intelligent_scoring(articles: list, user_alerts: list) -> list:
    """Apply intelligent relevance scoring using Gemini AI"""
    try:
        # Deduplicate articles by URL
        seen_urls = set()
        unique_articles = []
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)

        # Batch process articles for efficiency (smaller batches to avoid rate limits)
        enhanced_articles = []
        batch_size = 3

        for i in range(0, len(unique_articles), batch_size):
            batch = unique_articles[i:i + batch_size]
            enhanced_batch = await analyze_article_batch_intelligence(batch, user_alerts)
            enhanced_articles.extend(enhanced_batch)

        return enhanced_articles

    except Exception as e:
        logger.error(f"Error in intelligent scoring: {e}")
        return articles

async def analyze_article_batch_intelligence(articles: list, user_alerts: list) -> list:
    """Analyze a batch of articles using Gemini for intelligent relevance scoring"""
    try:
        if not articles:
            return articles

        # Prepare context about user alerts
        alerts_context = []
        for alert in user_alerts:
            alert_info = {
                "category": alert.get("main_category", ""),
                "keywords": alert.get("sub_categories", []),
                "followup_questions": alert.get("followup_questions", []),
                "custom_question": alert.get("custom_question", "")
            }
            alerts_context.append(alert_info)

        # Prepare articles for analysis
        articles_data = []
        for idx, article in enumerate(articles):
            article_data = {
                "id": idx,
                "title": article.get('title', '')[:200],
                "content": (article.get('content', '') or article.get('summary', '') or article.get('description', ''))[:500],
                "category": article.get('category', ''),
                "current_score": article.get('relevance_score', 0.5)
            }
            articles_data.append(article_data)

        # Create intelligent prompt for Gemini
        prompt = f"""
        Analyze these news articles for relevance to user alerts. For each article, provide a relevance score from 0.0 to 1.0 and brief reasoning.

        User Alerts Context:
        {json.dumps(alerts_context, indent=2)}

        Articles to Analyze:
        {json.dumps(articles_data, indent=2)}

        For each article, respond with JSON in this format:
        {{
            "article_id": <id>,
            "relevance_score": <0.0-1.0>,
            "reasoning": "<brief explanation>",
            "matched_criteria": ["<specific matches>"]
        }}

        Respond with a JSON array of analysis results.
        """

        # Call Gemini API with rate limiting
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Add delay to avoid rate limiting
            await asyncio.sleep(1)

            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                json={
                    'contents': [{'parts': [{'text': prompt}]}],
                    'generationConfig': {
                        'temperature': 0.3,
                        'maxOutputTokens': 1024
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                ai_response = result['candidates'][0]['content']['parts'][0]['text']

                # Parse AI response
                try:
                    # Extract JSON from response
                    json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                    if json_match:
                        analysis_results = json.loads(json_match.group())

                        # Apply AI scores to articles
                        for result in analysis_results:
                            article_idx = result.get('article_id')
                            if 0 <= article_idx < len(articles):
                                articles[article_idx]['ai_relevance_score'] = result.get('relevance_score', 0.5)
                                articles[article_idx]['ai_reasoning'] = result.get('reasoning', '')
                                articles[article_idx]['matched_criteria'] = result.get('matched_criteria', [])

                                # Calculate final score combining RAG and AI scores
                                rag_score = articles[article_idx].get('relevance_score', 0.5)
                                ai_score = result.get('relevance_score', 0.5)
                                articles[article_idx]['final_relevance_score'] = (rag_score * 0.4) + (ai_score * 0.6)

                except json.JSONDecodeError:
                    logger.warning("Could not parse AI response as JSON, using fallback scoring")
                    for article in articles:
                        article['final_relevance_score'] = article.get('relevance_score', 0.5)

            else:
                logger.warning(f"Gemini API error: {response.status_code}, using fallback scoring")
                for article in articles:
                    article['final_relevance_score'] = article.get('relevance_score', 0.5)

        return articles

    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        # Fallback to original scores
        for article in articles:
            article['final_relevance_score'] = article.get('relevance_score', 0.5)
        return articles

async def build_contextual_query(alert: dict, keywords: list, followup_questions: list, custom_question: str, category: str) -> str:
    """Build an intelligent contextual query that understands context and intent"""
    try:
        # Extract context clues from the alert
        query_parts = []
        _ = alert  # Parameter kept for future extensibility

        # Add category as primary context
        if category:
            query_parts.append(category)

        # Process keywords with context understanding
        if keywords:
            for keyword in keywords:
                keyword_lower = keyword.lower()

                # Detect specific tournaments/competitions
                if any(tournament in keyword_lower for tournament in ['asia cup', 'asiacup', 'world cup', 'ipl', 'premier league']):
                    query_parts.append(keyword)

                    # Add gender/age context if missing
                    if 'asia cup' in keyword_lower or 'asiacup' in keyword_lower:
                        # Default to men's unless specified
                        if not any(gender in keyword_lower for gender in ['women', 'mens', "men's", 'junior', 'u21', 'u19']):
                            query_parts.append("men's cricket")

                # Add sport-specific context
                elif keyword_lower in ['cricket', 'football', 'soccer', 'tennis', 'hockey']:
                    query_parts.append(keyword)

                    # For cricket, assume international unless specified
                    if keyword_lower == 'cricket':
                        query_parts.append("international cricket")

                else:
                    query_parts.append(keyword)

        # Process followup questions for intent
        if followup_questions:
            for question in followup_questions:
                question_lower = question.lower()

                # Extract specific intent indicators
                if 'win' in question_lower or 'victory' in question_lower:
                    query_parts.append("match results")
                elif 'final' in question_lower:
                    query_parts.append("final match")
                elif 'score' in question_lower:
                    query_parts.append("live scores")

                # Add the full question for context
                query_parts.append(question)

        # Process custom question with intelligence
        if custom_question and custom_question.strip():
            custom_lower = custom_question.lower()

            # Extract meaningful entities and intent
            meaningful_words = [word for word in custom_question.split() if len(word) > 3]
            query_parts.extend(meaningful_words)

            # Add intent understanding
            if any(word in custom_lower for word in ['notify', 'alert', 'tell', 'inform']):
                query_parts.append("breaking news")

        # Create contextual query
        contextual_query = " ".join(query_parts)

        logger.info(f"Built contextual query: {contextual_query}")
        return contextual_query

    except Exception as e:
        logger.error(f"Error building contextual query: {e}")
        # Fallback to simple concatenation
        simple_query = " ".join([category] + keywords + followup_questions + [custom_question])
        return simple_query.strip()

async def apply_intelligent_content_filtering(articles: list, alert_context: dict) -> list:
    """Apply smart filtering to remove irrelevant content"""
    _ = alert_context  # Not used anymore - simplified approach
    return articles  # Simplified - let spam filter handle filtering

def remove_obvious_spam(articles: list, user_alerts: list) -> list:
    """Remove obviously irrelevant articles based on user alert context"""
    try:
        if not articles or not user_alerts:
            return articles

        filtered_articles = []

        # Get user context
        user_keywords = []
        user_categories = []

        for alert in user_alerts:
            keywords = alert.get("sub_categories", [])
            category = alert.get("main_category", "")
            user_keywords.extend([k.lower() for k in keywords])
            if category:
                user_categories.append(category.lower())

        for article in articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower() + ' ' + article.get('summary', '').lower()
            article_category = article.get('category', '').lower()

            # Keep if relevance score is decent
            if article.get('relevance_score', 0) > 0.4:
                filtered_articles.append(article)
                continue

            # Basic relevance check
            keep_article = False

            # Check if article matches user's sport/category interests
            if 'cricket' in user_keywords:
                # Must have cricket-related terms
                if any(term in title + content for term in ['cricket', 'wicket', 'bowling', 'batting', 'asia cup', 'asiacup']):
                    keep_article = True
                # Exclude other sports unless specifically relevant
                elif any(term in title + content for term in ['hockey', 'football', 'rugby', 'tennis', 'golf', 'movies', 'film', 'cinema']):
                    keep_article = False
                    continue

            # Check category match
            if user_categories and article_category:
                if any(cat in article_category for cat in user_categories):
                    keep_article = True

            # If still unsure, keep articles with decent relevance
            if not keep_article and article.get('relevance_score', 0) > 0.25:
                keep_article = True

            if keep_article:
                filtered_articles.append(article)

        logger.info(f"Spam filter: {len(articles)} -> {len(filtered_articles)} articles")
        return filtered_articles

    except Exception as e:
        logger.error(f"Error in spam filtering: {e}")
        return articles

def remove_obvious_spam_per_alert(articles: list, alert: dict) -> list:
    """Strong ML/RAG-based filtering - reduces dependency on Gemini"""
    try:
        if not articles or not alert:
            return articles

        filtered_articles = []
        keywords = [k.lower() for k in alert.get("sub_categories", [])]
        category = alert.get("main_category", "").lower()
        followup_questions = [q.lower() for q in alert.get("followup_questions", [])]
        custom_question = alert.get("custom_question", "").lower()

        # Build comprehensive user intent profile
        all_user_text = ' '.join(keywords + followup_questions + [custom_question])
        user_intent_keywords = set(word for word in all_user_text.split() if len(word) > 3)

        # If no user intent, accept all articles from same category
        if not user_intent_keywords:
            return articles

        for article in articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower() + ' ' + article.get('summary', '').lower() + ' ' + article.get('description', '').lower()
            article_text = title + ' ' + content
            article_category = article.get('category', '').lower()

            # STEP 1: Category must match (basic filter)
            if article_category != category:
                continue

            # STEP 2: ML/RAG Relevance Score (primary signal)
            relevance_score = article.get('relevance_score', 0)

            # STEP 3: Keyword overlap analysis (semantic matching)
            matched_keywords = [kw for kw in user_intent_keywords if kw in article_text]
            keyword_match_count = len(matched_keywords)
            keyword_match_ratio = keyword_match_count / len(user_intent_keywords) if user_intent_keywords else 0

            # STEP 4: Title relevance (titles are usually more focused)
            title_keyword_matches = sum(1 for kw in user_intent_keywords if kw in title)

            # STEP 5: Strong ML-based filtering logic
            # Prioritize RAG relevance score but validate with keyword matching
            keep_article = False

            # High confidence: Strong RAG score + keyword matches
            if relevance_score > 0.5 and keyword_match_count >= 2:
                keep_article = True
            # Medium-high confidence: Good RAG score + some keyword matches
            elif relevance_score > 0.4 and keyword_match_count >= 1:
                keep_article = True
            # Title match: Keywords in title are strong signals
            elif title_keyword_matches >= 2:
                keep_article = True
            # Moderate confidence: Decent overlap ratio
            elif keyword_match_ratio >= 0.4 and relevance_score > 0.3:
                keep_article = True
            # Strong keyword presence even if lower RAG score
            elif keyword_match_count >= 3:
                keep_article = True

            if keep_article:
                # Add match metadata for Gemini to use
                article['ml_keyword_matches'] = matched_keywords
                article['ml_match_ratio'] = keyword_match_ratio
                article['ml_confidence'] = relevance_score
                filtered_articles.append(article)

        logger.info(f"ML/RAG filter for {category}: {len(articles)} -> {len(filtered_articles)} articles (RAG score + keyword matching)")
        return filtered_articles

    except Exception as e:
        logger.error(f"Error in ML/RAG filtering: {e}")
        return articles

async def send_wati_notification(user_id: str, alert_result: dict) -> dict:
    """Send WhatsApp notification via WATI API with contact registration fallback"""
    try:
        print(f"🔍 WATI DEBUG: Starting WhatsApp notification for user_id: {user_id}")

        # Get user's WhatsApp number from database - try multiple sources
        whatsapp_number = None

        # Try 1: Users collection
        print(f"🔍 WATI DEBUG: Checking users collection for user_id: {user_id}")
        user_doc = await db.get_collection("users").find_one({"user_id": user_id})
        print(f"🔍 WATI DEBUG: User doc found: {user_doc is not None}")
        if user_doc:
            print(f"🔍 WATI DEBUG: User doc keys: {list(user_doc.keys())}")
            # Check for direct whatsapp_number field
            if user_doc.get("whatsapp_number"):
                whatsapp_number = user_doc.get("whatsapp_number")
                print(f"🔍 WATI DEBUG: Found WhatsApp number in users collection: {whatsapp_number}")
            # Check for country_code + phone_number combination
            elif user_doc.get("country_code") and user_doc.get("phone_number"):
                country_code = user_doc.get("country_code")
                phone_number = user_doc.get("phone_number")
                whatsapp_number = f"{country_code}{phone_number}"
                print(f"🔍 WATI DEBUG: Constructed WhatsApp number from country_code + phone_number: {whatsapp_number}")
            # Check for just phone_number (might include country code)
            elif user_doc.get("phone_number"):
                phone_number = user_doc.get("phone_number")
                # Add +91 if phone number doesn't start with +
                if not phone_number.startswith('+'):
                    whatsapp_number = f"+91{phone_number}"
                else:
                    whatsapp_number = phone_number
                print(f"🔍 WATI DEBUG: Found phone_number in users collection: {whatsapp_number}")

        # Try 2: User alerts collection (might have phone number)
        if not whatsapp_number:
            print(f"🔍 WATI DEBUG: Checking alerts collection for user_id: {user_id}")
            alert_doc = await db.get_collection("alerts").find_one({"user_id": user_id})
            print(f"🔍 WATI DEBUG: Alert doc found: {alert_doc is not None}")
            if alert_doc:
                print(f"🔍 WATI DEBUG: Alert doc keys: {list(alert_doc.keys())}")
                if alert_doc.get("phone_number"):
                    whatsapp_number = alert_doc.get("phone_number")
                    print(f"🔍 WATI DEBUG: Found phone_number in alerts collection: {whatsapp_number}")
                elif alert_doc.get("whatsapp_number"):
                    whatsapp_number = alert_doc.get("whatsapp_number")
                    print(f"🔍 WATI DEBUG: Found whatsapp_number in alerts collection: {whatsapp_number}")

        # Try 3: Check if user_id itself looks like a phone number
        if not whatsapp_number and user_id.isdigit() and len(user_id) >= 10:
            whatsapp_number = user_id
            print(f"🔍 WATI DEBUG: Using user_id as phone number: {whatsapp_number}")

        if not whatsapp_number:
            print(f"❌ WATI DEBUG: No WhatsApp number found for user {user_id}")
            return {"status": "error", "message": "User WhatsApp number not found in any collection"}

        # Get the best article for the alert
        articles = alert_result.get('articles', [])
        print(f"🔍 WATI DEBUG: Found {len(articles)} articles for notification")
        if not articles:
            print(f"❌ WATI DEBUG: No articles to send for user {user_id}")
            return {"status": "skipped", "message": "No articles to send"}

        best_article = articles[0]  # Get the highest relevance article
        print(f"🔍 WATI DEBUG: Selected best article: {best_article.get('title', 'No title')[:50]}...")

        # Prepare template parameters
        # {{1}} = image URL
        # {{2}} = title
        # {{3}} = description

        image_url = best_article.get('image_url', '') or best_article.get('enhanced_image_url', '')
        title = best_article.get('enhanced_title', '') or best_article.get('title', '')
        print(f"🔍 WATI DEBUG: Template params - Image URL: {image_url[:50]}..., Title: {title[:50]}...")

        # Get full description - no truncation
        description = best_article.get('short_description', '')
        if not description:
            # Fallback to full summary/content - send complete text
            description = best_article.get('summary', '') or best_article.get('content', '') or best_article.get('description', '')

        # If still empty, use a default message
        if not description:
            description = "Read full article for details."

        print(f"🔍 WATI DEBUG: Final description length: {len(description)} chars")

        print(f"🔍 WATI DEBUG: WATI payload prepared with template: {WATI_TEMPLATE_NAME}")

        # Send WhatsApp message via WATI
        headers = {
            'Authorization': WATI_ACCESS_TOKEN,
            'Content-Type': 'application/json-patch+json'
        }

        # Use correct WATI API format with receivers array
        final_payload = {
            "receivers": [
                {
                    "whatsappNumber": whatsapp_number,
                    "customParams": [
                        {
                            "name": "1",
                            "value": image_url
                        },
                        {
                            "name": "2",
                            "value": title
                        },
                        {
                            "name": "3",
                            "value": description
                        }
                    ]
                }
            ],
            "template_name": WATI_TEMPLATE_NAME,
            "broadcast_name": WATI_BROADCAST_NAME
        }

        # Alternative payload format for testing - comment out the above and try this if needed
        # final_payload = {
        #     "whatsappNumber": whatsapp_number,
        #     "templateName": WATI_TEMPLATE_NAME,
        #     "broadcastName": WATI_BROADCAST_NAME,  # Remove alert category from broadcast name
        #     "templateData": {
        #         "1": image_url,
        #         "2": title,
        #         "3": description
        #     }
        # }

        # Use correct WATI endpoint - sendTemplateMessages (plural)
        api_endpoint = f"{WATI_BASE_URL}/api/v1/sendTemplateMessages"
        print(f"🔍 WATI DEBUG: Sending API request to: {api_endpoint}")
        print(f"🔍 WATI DEBUG: WhatsApp number: {whatsapp_number}")
        print(f"🔍 WATI DEBUG: WhatsApp number length: {len(whatsapp_number)}")
        print(f"🔍 WATI DEBUG: WhatsApp number format check: starts with +: {whatsapp_number.startswith('+')}")
        print(f"🔍 WATI DEBUG: Headers: Authorization present: {bool(headers.get('Authorization'))}")
        print(f"🔍 WATI DEBUG: Final payload keys: {list(final_payload.keys())}")
        print(f"🔍 WATI DEBUG: Complete payload: {final_payload}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    api_endpoint,
                    headers=headers,
                    json=final_payload
                )

                print(f"🔍 WATI DEBUG: API Response Status: {response.status_code}")
                print(f"🔍 WATI DEBUG: API Response Headers: {dict(response.headers)}")
                print(f"🔍 WATI DEBUG: API Response Text: {response.text}")

                # Parse response JSON to check WATI-specific success/error format
                try:
                    response_data = response.json()
                    print(f"🔍 WATI DEBUG: Parsed Response Data: {response_data}")

                    # Check WATI response format: {"result": true/false, "errors": {...}}
                    wati_result = response_data.get("result", False)
                    wati_errors = response_data.get("errors", {})

                    if response.status_code == 200 and wati_result:
                        print(f"✅ WATI DEBUG: WhatsApp notification sent successfully to {whatsapp_number}")
                        logger.info(f"WhatsApp notification sent successfully to {whatsapp_number}")
                        return {
                            "status": "success",
                            "whatsapp_number": whatsapp_number,
                            "template_used": WATI_TEMPLATE_NAME,
                            "article_title": title,
                            "message": "Notification sent successfully",
                            "wati_response": response_data
                        }
                    else:
                        error_msg = wati_errors.get("error", "Unknown WATI error")
                        invalid_numbers = wati_errors.get("invalidWhatsappNumbers", [])
                        invalid_params = wati_errors.get("invalidCustomParameters", [])

                        print(f"❌ WATI DEBUG: WATI API Error - Result: {wati_result}")
                        print(f"❌ WATI DEBUG: Error message: {error_msg}")
                        print(f"❌ WATI DEBUG: Invalid WhatsApp numbers: {invalid_numbers}")
                        print(f"❌ WATI DEBUG: Invalid parameters: {invalid_params}")

                        logger.error(f"WATI API error: {error_msg}")
                        return {
                            "status": "error",
                            "message": f"WATI API error: {error_msg}",
                            "invalid_numbers": invalid_numbers,
                            "invalid_params": invalid_params,
                            "response": response_data
                        }

                except json.JSONDecodeError:
                    print(f"❌ WATI DEBUG: Could not parse response as JSON")
                    print(f"❌ WATI DEBUG: Raw response: {response.text}")
                    logger.error(f"WATI API error: {response.status_code} - {response.text}")
                    return {
                        "status": "error",
                        "message": f"WATI API error: {response.status_code}",
                        "response": response.text
                    }
            except Exception as api_error:
                print(f"❌ WATI DEBUG: API Call Exception: {api_error}")
                raise api_error

    except Exception as e:
        logger.error(f"Error sending WATI notification: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

async def final_gemini_perfect_filter(articles: list, user_alerts: list, target_count: int) -> list:
    """Use Gemini to select only the most perfectly relevant articles"""
    try:
        if not articles or not user_alerts:
            return articles[:target_count]

        # Prepare user context
        user_context = {}
        for alert in user_alerts:
            user_context = {
                'category': alert.get('main_category', ''),
                'keywords': alert.get('sub_categories', []),
                'followup_questions': alert.get('followup_questions', []),
                'custom_question': alert.get('custom_question', ''),
                'intent': f"User wants {alert.get('main_category', '')} news about {', '.join(alert.get('sub_categories', []))}"
            }

        # Prepare articles for analysis with ML metadata
        articles_for_analysis = []
        for idx, article in enumerate(articles[:15]):  # Analyze top 15 to find best
            article_data = {
                'id': idx,
                'title': article.get('title', ''),
                'content': (article.get('content', '') or article.get('summary', '') or article.get('description', ''))[:400],
                'url': article.get('url', ''),
                'ml_relevance_score': article.get('relevance_score', 0),
                'ml_keyword_matches': article.get('ml_keyword_matches', []),
                'ml_match_ratio': article.get('ml_match_ratio', 0),
                'ml_confidence': article.get('ml_confidence', 0)
            }
            articles_for_analysis.append(article_data)

        # Extract user context dynamically
        user_keywords = user_context.get('keywords', [])
        user_questions = user_context.get('followup_questions', [])
        user_custom = user_context.get('custom_question', '')
        user_category = user_context.get('category', '')

        # Detect gender/demographic context
        all_user_text = ' '.join(user_keywords + user_questions + [user_custom]).lower()
        is_womens_context = any(term in all_user_text for term in ["women's", 'womens', 'women', 'female', 'ladies'])
        gender_context = "Women's" if is_womens_context else "Men's/General"

        # Create dynamic filtering rules based on user's actual input
        user_specific_terms = set()
        for text in user_keywords + user_questions + [user_custom]:
            user_specific_terms.update([word for word in text.lower().split() if len(word) > 3])

        # Create AI prompt that leverages ML/RAG scores
        perfect_prompt = f"""
        You are the final validation layer. These articles are PRE-FILTERED by ML/RAG models.

        USER'S REQUEST:
        Category: {user_category}
        Keywords: {user_keywords}
        Questions: {user_questions}
        Custom Question: "{user_custom}"

        ML-FILTERED ARTICLES:
        {json.dumps(articles_for_analysis, indent=2)}

        ML/RAG SIGNALS EXPLAINED:
        - ml_relevance_score: Semantic similarity (0-1) from embedding models
        - ml_keyword_matches: User keywords found in article
        - ml_match_ratio: % of user keywords matched
        - ml_confidence: Overall ML confidence

        YOUR ROLE (Final Semantic Validation):
        1. ML has already filtered - trust the ml_relevance_score
        2. Check if ML matches align with user's TRUE context
        3. Select BEST {target_count} article(s) that will satisfy user

        VALIDATION CHECKS:
        - Does context match? (If user mentions "Rahul", is article about male cricket?)
        - Does format match? (Test vs ODI, Bollywood vs Hollywood, etc.)
        - Will this answer: "{user_custom}"?
        - Are ml_keyword_matches actually relevant?

        RESPONSE ([] if nothing meets 85%+):
        [
            {{
                "article_id": <id>,
                "enhanced_title": "<60 char>",
                "short_description": "<2-3 lines>",
                "satisfaction_reason": "<using ML scores + context>",
                "predicted_satisfaction": "XX%",
                "engagement_factors": ["<from ML + validation>"]
            }}
        ]

        CRITICAL:
        - ML did heavy lifting - validate context/semantics
        - Return [] if no perfect match
        - Max {target_count} articles
        """

        # Call Gemini for perfect filtering
        async with httpx.AsyncClient(timeout=25.0) as client:
            # Add small delay for rate limiting
            await asyncio.sleep(0.5)

            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                json={
                    'contents': [{'parts': [{'text': perfect_prompt}]}],
                    'generationConfig': {
                        'temperature': 0.2,
                        'maxOutputTokens': 1000
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                ai_response = result['candidates'][0]['content']['parts'][0]['text']

                try:
                    # Extract JSON from response
                    json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                    if json_match:
                        perfect_results = json.loads(json_match.group())

                        # Get ultra-enhanced articles with psychological intelligence
                        perfect_articles = []
                        for selection in perfect_results:
                            article_idx = selection.get('article_id')
                            enhanced_title = selection.get('enhanced_title', '')
                            short_description = selection.get('short_description', '')
                            satisfaction_reason = selection.get('satisfaction_reason', '')
                            predicted_satisfaction = selection.get('predicted_satisfaction', 'GOOD - 80%')
                            engagement_factors = selection.get('engagement_factors', [])

                            if 0 <= article_idx < len(articles):
                                selected_article = articles[article_idx]

                                # Add elite-level intelligent enhancements
                                selected_article['enhanced_title'] = enhanced_title
                                selected_article['short_description'] = short_description
                                selected_article['satisfaction_reason'] = satisfaction_reason
                                selected_article['predicted_satisfaction'] = predicted_satisfaction
                                selected_article['engagement_factors'] = engagement_factors
                                selected_article['perfect_match'] = True
                                selected_article['curation_quality'] = 'ELITE_AI_SELECTED'
                                selected_article['user_satisfaction_score'] = 10
                                # image_url already exists from RSS extraction

                                perfect_articles.append(selected_article)

                        logger.info(f"Gemini perfect filter: {len(articles)} -> {len(perfect_articles)} articles")
                        return perfect_articles[:target_count]

                except json.JSONDecodeError:
                    logger.warning("Could not parse Gemini perfect filter response")

            else:
                logger.warning(f"Gemini perfect filter API error: {response.status_code}")

        # Fallback: return top articles by relevance score
        articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        logger.info(f"Fallback perfect filter: returning top {target_count} articles")
        return articles[:target_count]

    except Exception as e:
        logger.error(f"Error in perfect filtering: {e}")
        return articles[:target_count]

async def get_alert_specific_articles(alert: dict, alert_query: str, category: str) -> list:
    """Get articles specific to this alert without mixing with other alerts"""
    try:
        _ = alert  # Parameter kept for future extensibility
        logger.info(f"Fetching alert-specific articles for category: {category}")

        if category not in CATEGORY_RSS_FEEDS:
            return []

        # Step 1: Fetch fresh RSS articles for this specific category
        category_feeds = CATEGORY_RSS_FEEDS[category]
        category_articles = []

        for feed_url in category_feeds:
            feed_articles = await fetch_rss_feed(feed_url)
            for article in feed_articles:
                article['category'] = category
                category_articles.append(article)

        if not category_articles:
            return []

        # Step 2: Use existing RAG system's embedding model (memory efficient)
        # Reuse the same model instead of loading a new one
        model = rag_system.embedding_model

        # Generate query embedding
        query_embedding = model.encode(alert_query)

        # Filter articles from last 24 hours only
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=24)

        recent_articles = []
        for article in category_articles:
            try:
                # Parse published date
                published_date_str = article.get('published_date', '')
                if published_date_str:
                    # Handle different date formats
                    try:
                        # Format: "Sun, 28 Sep 2025 14:05:56 +0530"
                        from email.utils import parsedate_to_datetime
                        published_date = parsedate_to_datetime(published_date_str)

                        # Check if article is within last 24 hours
                        if published_date >= cutoff_time:
                            recent_articles.append(article)
                    except:
                        # If date parsing fails, include the article (better to show than miss)
                        recent_articles.append(article)
                else:
                    # If no date, include the article
                    recent_articles.append(article)
            except:
                # If any error, include the article
                recent_articles.append(article)

        logger.info(f"24-hour filter: {len(category_articles)} -> {len(recent_articles)} recent articles")

        # Generate embeddings for recent articles and calculate similarity
        scored_articles = []
        for article in recent_articles:
            article_text = f"{article.get('title', '')} {article.get('content', '')} {article.get('summary', '')}"
            article_embedding = model.encode(article_text)

            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(query_embedding, article_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding)
            )

            article['relevance_score'] = float(similarity)
            if similarity > 0.1:  # Lowered threshold to catch more relevant articles
                scored_articles.append(article)

        # Step 3: Sort by relevance and return top articles
        scored_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        logger.info(f"Alert-specific filtering: {len(category_articles)} -> {len(scored_articles)} relevant articles")
        return scored_articles[:15]  # Return top 15 relevant articles

    except Exception as e:
        logger.error(f"Error in alert-specific article retrieval: {e}")
        return []

async def debug_user_satisfaction(articles: list, user_alerts: list) -> dict:
    """Debug function to check if articles will satisfy user requirements"""
    try:
        if not articles or not user_alerts:
            return {"status": "no_data", "message": "No articles or alerts to check"}

        # Get user requirements
        user_requirements = {}
        for alert in user_alerts:
            user_requirements = {
                'main_category': alert.get('main_category', '').lower(),
                'keywords': [k.lower() for k in alert.get('sub_categories', [])],
                'followup_questions': [q.lower() for q in alert.get('followup_questions', [])],
                'custom_question': alert.get('custom_question', '').lower()
            }

        # Analyze each article
        satisfaction_analysis = []
        overall_satisfaction = 0

        for idx, article in enumerate(articles):
            title = article.get('title', '').lower()
            content = (article.get('content', '') + ' ' + article.get('summary', '')).lower()

            # Check satisfaction criteria
            matches = {
                'category_match': False,
                'keyword_match': False,
                'followup_match': False,
                'custom_match': False
            }

            # Category check - Fixed to work for all categories
            user_category = user_requirements['main_category']
            article_category = article.get('category', '').lower()
            if user_category == article_category:
                matches['category_match'] = True

            # Keywords check
            if user_requirements['keywords']:
                for keyword in user_requirements['keywords']:
                    if keyword in title or keyword in content:
                        matches['keyword_match'] = True
                        break

            # Followup questions check
            if user_requirements['followup_questions']:
                for question in user_requirements['followup_questions']:
                    if question in title or question in content:
                        matches['followup_match'] = True
                        break

            # Custom question check
            if user_requirements['custom_question']:
                custom_words = user_requirements['custom_question'].split()
                if any(word in title or word in content for word in custom_words if len(word) > 3):
                    matches['custom_match'] = True

            # Calculate satisfaction score
            score = sum(matches.values()) / len(matches) * 100
            overall_satisfaction += score

            satisfaction_analysis.append({
                'article_id': idx,
                'title': article.get('title', '')[:50] + '...',
                'satisfaction_score': round(score, 1),
                'matches': matches,
                'has_image': bool(article.get('image_url', '')),
                'relevance_score': article.get('relevance_score', 0)
            })

        overall_satisfaction = overall_satisfaction / len(articles) if articles else 0

        # Determine satisfaction level
        if overall_satisfaction >= 75:
            satisfaction_level = "EXCELLENT - User will be very satisfied"
        elif overall_satisfaction >= 50:
            satisfaction_level = "GOOD - User will be satisfied"
        elif overall_satisfaction >= 25:
            satisfaction_level = "FAIR - User might be satisfied"
        else:
            satisfaction_level = "POOR - User may not be satisfied"

        debug_report = {
            'overall_satisfaction_score': round(overall_satisfaction, 1),
            'satisfaction_level': satisfaction_level,
            'user_requirements': user_requirements,
            'articles_analysis': satisfaction_analysis,
            'total_articles': len(articles),
            'recommendation': "These articles should satisfy user intent" if overall_satisfaction >= 50 else "Consider improving article relevance"
        }

        logger.info(f"Satisfaction Debug: {overall_satisfaction:.1f}% - {satisfaction_level}")
        return debug_report

    except Exception as e:
        logger.error(f"Error in satisfaction debug: {e}")
        return {"status": "error", "message": str(e)}

async def update_user_profile_from_alerts(user_id: str, user_alerts: list, relevant_articles: list):
    """Learn from user alert patterns and update user profile intelligently"""
    try:
        # Extract interests and preferences from alerts
        interests = set()
        preferred_categories = set()
        preferred_sources = set()

        for alert in user_alerts:
            # Extract category preferences
            category = alert.get("main_category", "")
            if category:
                preferred_categories.add(category.lower())

            # Extract keyword interests
            keywords = alert.get("sub_categories", [])
            interests.update([kw.lower() for kw in keywords if kw])

            # Extract interests from followup questions
            followup_questions = alert.get("followup_questions", [])
            for question in followup_questions:
                # Extract meaningful words (> 3 characters)
                words = [word.lower() for word in question.split() if len(word) > 3]
                interests.update(words)

            # Extract interests from custom questions
            custom_question = alert.get("custom_question", "")
            if custom_question:
                words = [word.lower() for word in custom_question.split() if len(word) > 3]
                interests.update(words)

        # Learn from articles that scored well
        for article in relevant_articles:
            if article.get('final_relevance_score', 0) > 0.7:  # High relevance threshold
                source = article.get('source', '')
                if source:
                    preferred_sources.add(source)

        # Update user profile in database using RAG system
        profile_update = {
            'interests': list(interests),
            'preferred_categories': list(preferred_categories),
            'preferred_sources': list(preferred_sources),
            'last_alert_update': datetime.now(timezone.utc),
            'alert_count': len(user_alerts)
        }

        # Use the existing RAG system's personalization engine
        await rag_system.personalization_engine.update_user_profile(user_id, {
            'profile_data': profile_update
        })

        logger.info(f"Updated profile for user {user_id}: {len(interests)} interests, {len(preferred_categories)} categories")

    except Exception as e:
        logger.error(f"Error updating user profile from alerts: {e}")

# Auto news refresh function (called when needed)
async def refresh_news_if_needed():
    """Refresh news data if the database is empty or data is old"""
    try:
        doc_count = await db.get_collection("document_vectors").count_documents({})
        if doc_count == 0:
            logger.info("No documents found, fetching fresh news...")
            articles = await fetch_all_news()
            if articles:
                await rag_system.process_news_articles(articles)
                logger.info(f"Fetched and processed {len(articles)} articles")
    except Exception as e:
        logger.error(f"Error in news refresh: {e}")