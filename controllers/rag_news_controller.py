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

from core.rag_system import RAGSystem
from db.mongo import db  # Use existing MongoDB connection

# Setup logging
logger = logging.getLogger(__name__)

# API Configuration
GEMINI_API_KEY = 'AIzaSyCpPZ2xRGJGx06tZFw_XfU_RTsBiIF_afg'

# Initialize RAG System
rag_system = RAGSystem(db, GEMINI_API_KEY)

# FastAPI Router
router = APIRouter()

# RSS Feeds Configuration
CATEGORY_RSS_FEEDS = {
    "sports": [
        "https://www.thehindu.com/sport/feeder/default.rss",
        "https://feeds.bbci.co.uk/sport/rss.xml",
        "https://rss.cnn.com/rss/edition_sport.rss"
    ],
    "news": [
        "https://feeds.bbci.co.uk/news/rss.xml",
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

        # Step 2: Process each alert and get relevant news using RAG
        all_relevant_news = []

        for alert in user_alerts:
            alert_id = alert.get("_id")
            category = alert.get("main_category", "").lower()
            keywords = alert.get("sub_categories", [])
            followup_questions = alert.get("followup_questions", [])
            custom_question = alert.get("custom_question", "")

            logger.info(f"Processing alert {alert_id}: category={category}, keywords={keywords}")

            # Step 3: Build intelligent contextual query from alert data
            alert_query = await build_contextual_query(alert, keywords, followup_questions, custom_question, category)

            # Step 4: Use RAG system for intelligent article retrieval
            relevant_articles = await rag_system.retrieve_personalized_news(
                user_id=user_id,
                query=alert_query,
                limit=20  # Get more articles for this alert
            )

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

            # Then enhance with alert context
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

                all_relevant_news.append(article)

        # Step 6: Remove obvious irrelevant articles first
        all_relevant_news = remove_obvious_spam(all_relevant_news, user_alerts)

        # Step 7: Final intelligent filtering with Gemini to get perfect results (max 3 articles)
        if all_relevant_news:
            all_relevant_news = await final_gemini_perfect_filter(all_relevant_news, user_alerts, 3)

        # Step 8: Final satisfaction debug check
        satisfaction_report = await debug_user_satisfaction(all_relevant_news, user_alerts)

        # Step 9: Learn from user alert patterns and update profile
        await update_user_profile_from_alerts(user_id, user_alerts, all_relevant_news)

        return {
            "status": "success",
            "user_id": user_id,
            "alerts_processed": len(user_alerts),
            "total_articles": len(all_relevant_news),
            "articles": all_relevant_news,
            "satisfaction_debug": satisfaction_report,
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

        # Prepare articles for analysis
        articles_for_analysis = []
        for idx, article in enumerate(articles[:15]):  # Analyze top 15 to find best 2-3
            article_data = {
                'id': idx,
                'title': article.get('title', ''),
                'content': (article.get('content', '') or article.get('summary', '') or article.get('description', ''))[:400],
                'url': article.get('url', ''),
                'relevance_score': article.get('relevance_score', 0)
            }
            articles_for_analysis.append(article_data)

        # Create comprehensive intelligent prompt that does everything in one call
        perfect_prompt = f"""
        You are an expert news curator. Analyze user requirements and select ONLY the {target_count} most satisfying articles.

        USER PROFILE ANALYSIS:
        - Main Category: {user_context.get('category', '')}
        - Sub-Categories/Keywords: {user_context.get('keywords', [])}
        - Follow-up Questions: {user_context.get('followup_questions', [])}
        - Custom Question: {user_context.get('custom_question', '')}
        - User Intent: {user_context.get('intent', '')}

        AVAILABLE ARTICLES:
        {json.dumps(articles_for_analysis, indent=2)}

        TASK: Select {target_count} articles that will make the user happy and satisfied. For each selected article, provide:
        1. Enhanced title (engaging, clickable)
        2. Short description (2-3 lines that hook the user)
        3. Satisfaction reason (why user will love this)

        Note: Images will be taken directly from RSS feeds, no need to suggest images.

        SELECTION CRITERIA:
        ✅ Perfectly matches main category + sub-categories
        ✅ Addresses follow-up questions and custom questions
        ✅ Recent and valuable content
        ✅ Will genuinely interest and satisfy the user
        ✅ No duplicate or similar content

        RESPOND WITH JSON:
        [
            {{
                "article_id": 0,
                "enhanced_title": "Engaging title that makes user want to click",
                "short_description": "2-3 lines that perfectly explain why this news matters to the user and hooks their interest",
                "satisfaction_reason": "Specific reason why this article perfectly satisfies user's intent and will make them happy"
            }}
        ]

        Select maximum {target_count} articles that will truly satisfy the user!
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

                        # Get enhanced articles with all the intelligent additions
                        perfect_articles = []
                        for selection in perfect_results:
                            article_idx = selection.get('article_id')
                            enhanced_title = selection.get('enhanced_title', '')
                            short_description = selection.get('short_description', '')
                            satisfaction_reason = selection.get('satisfaction_reason', '')

                            if 0 <= article_idx < len(articles):
                                selected_article = articles[article_idx]

                                # Add intelligent enhancements (image comes from RSS)
                                selected_article['enhanced_title'] = enhanced_title
                                selected_article['short_description'] = short_description
                                selected_article['satisfaction_reason'] = satisfaction_reason
                                selected_article['perfect_match'] = True
                                selected_article['user_satisfaction_score'] = 10  # Perfect match
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

            # Category check
            if user_requirements['main_category'] == 'sports' and article.get('category', '').lower() == 'sports':
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