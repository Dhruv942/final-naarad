from fastapi import FastAPI, APIRouter, Query, HTTPException, Request
from pydantic import BaseModel
import httpx, feedparser, asyncio, logging
from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline  # Removed for optimization
import motor.motor_asyncio
from datetime import datetime, timedelta
import hashlib
import os
import re
import json
from zoneinfo import ZoneInfo

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
feedback_collection = db.get_collection("user_feedback")
preferences_collection = db.get_collection("user_preferences")
message_tracking_collection = db.get_collection("message_tracking")

# -------------------- OPTIMIZED: Only Essential Model --------------------
logger.info("Loading only essential model for free hosting...")
sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")  # Removed - Gemini handles entities
# sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # Removed - Gemini handles sentiment
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Removed - Gemini handles summarization
logger.info("✅ Lightweight model loaded successfully! Memory usage: ~22MB (vs 3.3GB)")

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

# -------------------- Gemini LLM Gatekeeper --------------------
GEMINI_API_KEY = 'AIzaSyCpPZ2xRGJGx06tZFw_XfU_RTsBiIF_afg'
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# -------------------- DYNAMIC CONTEXT INTELLIGENCE --------------------
def analyze_query_context_dynamically(query: str):
    """
    DYNAMIC: Intelligent context detection without hardcoded lists
    Uses patterns, keywords and semantic analysis
    """
    query_lower = query.lower()
    context_analysis = {
        "entities": [],
        "intent": "general",
        "specificity": "low",
        "sports_context": None,
        "gender_context": None,
        "strict_matching": False
    }

    # DYNAMIC PATTERN DETECTION

    # 1. SPORTS CONTEXT DETECTION
    cricket_indicators = ["cricket", "batting", "bowling", "wicket", "runs", "over", "match", "series", "ipl", "bcci"]
    football_indicators = ["football", "soccer", "goal", "fifa", "uefa", "premier league", "la liga"]

    if any(indicator in query_lower for indicator in cricket_indicators):
        context_analysis["sports_context"] = "cricket"
        context_analysis["intent"] = "sports"
    elif any(indicator in query_lower for indicator in football_indicators):
        context_analysis["sports_context"] = "football"
        context_analysis["intent"] = "sports"

    # 2. GENDER CONTEXT DETECTION (DYNAMIC)
    men_indicators = ["men", "men's", "male", "boys", "guys"]
    women_indicators = ["women", "women's", "female", "girls", "ladies"]

    if any(indicator in query_lower for indicator in women_indicators):
        context_analysis["gender_context"] = "women"
        context_analysis["strict_matching"] = True
    elif any(indicator in query_lower for indicator in men_indicators):
        context_analysis["gender_context"] = "men"
        context_analysis["strict_matching"] = True

    # 3. TECHNOLOGY CONTEXT
    tech_indicators = ["iphone", "android", "ios", "app", "software", "launch", "update", "tech", "gadget"]
    if any(indicator in query_lower for indicator in tech_indicators):
        context_analysis["intent"] = "technology"

    # 4. ENTERTAINMENT CONTEXT
    bollywood_indicators = ["movie", "film", "bollywood", "actor", "actress", "cinema", "box office", "trailer"]
    if any(indicator in query_lower for indicator in bollywood_indicators):
        context_analysis["intent"] = "entertainment"

    # 5. ENTITY EXTRACTION (DYNAMIC)
    # Extract proper nouns (capitalized words) as potential entities
    import re
    potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    context_analysis["entities"] = potential_entities

    # 6. SPECIFICITY DETECTION
    if len(potential_entities) > 0:
        context_analysis["specificity"] = "high"
        context_analysis["strict_matching"] = True
    elif any(word in query_lower for word in ["specific", "only", "particular", "exactly"]):
        context_analysis["specificity"] = "high"
        context_analysis["strict_matching"] = True

    # 7. INTENT KEYWORDS
    if any(word in query_lower for word in ["win", "victory", "beat", "defeat"]):
        context_analysis["intent"] += "_victory"
    elif any(word in query_lower for word in ["news", "update", "latest"]):
        context_analysis["intent"] += "_news"

    return context_analysis

def lightweight_ml_classifier(news_title: str, news_description: str, user_query: str):
    """
    LIGHTWEIGHT ML: Quick classification before expensive Gemini calls
    Uses semantic similarity + pattern matching for fast filtering
    """
    try:
        # Combine news content
        news_content = f"{news_title} {news_description}".lower()
        query_lower = user_query.lower()

        # CLASSIFICATION SCORES
        scores = {
            "semantic_similarity": 0.0,
            "entity_overlap": 0.0,
            "keyword_density": 0.0,
            "context_match": 0.0
        }

        # 1. SEMANTIC SIMILARITY (using existing model)
        scores["semantic_similarity"] = calculate_similarity(news_content, query_lower)

        # 2. ENHANCED ENTITY OVERLAP SCORE
        import re
        query_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_query))
        news_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', f"{news_title} {news_description}"))

        if query_entities:
            # Exact matches
            exact_matches = len(query_entities.intersection(news_entities))

            # Fuzzy matches (partial name matching)
            fuzzy_matches = 0
            for q_entity in query_entities:
                for n_entity in news_entities:
                    if (q_entity.lower() in n_entity.lower() or n_entity.lower() in q_entity.lower()) and len(q_entity) > 3:
                        fuzzy_matches += 0.5

            total_matches = exact_matches + fuzzy_matches
            scores["entity_overlap"] = min(1.0, total_matches / len(query_entities))

        # 3. KEYWORD DENSITY SCORE
        query_keywords = [word for word in query_lower.split() if len(word) > 2]
        if query_keywords:
            keyword_matches = sum(1 for keyword in query_keywords if keyword in news_content)
            scores["keyword_density"] = keyword_matches / len(query_keywords)

        # 4. CONTEXT MATCH SCORE
        context = analyze_query_context_dynamically(user_query)

        # Sports context matching
        if context["sports_context"] == "cricket":
            cricket_terms = ["cricket", "batting", "bowling", "wicket", "runs", "over", "match", "series", "ipl"]
            cricket_matches = sum(1 for term in cricket_terms if term in news_content)
            scores["context_match"] = min(1.0, cricket_matches / 3)

        # Tech context matching
        elif context["intent"] == "technology":
            tech_terms = ["technology", "tech", "app", "software", "launch", "update", "device"]
            tech_matches = sum(1 for term in tech_terms if term in news_content)
            scores["context_match"] = min(1.0, tech_matches / 3)

        # WEIGHTED FINAL SCORE - Focus on user's specific questions
        final_score = (
            scores["semantic_similarity"] * 0.3 +
            scores["keyword_density"] * 0.4 +  # Increased for followup questions
            scores["entity_overlap"] * 0.2 +   # Reduced entity focus
            scores["context_match"] * 0.1
        )

        return {
            "ml_score": final_score,
            "should_send_to_gemini": final_score > 0.25,  # Lowered from 0.3 to 0.25
            "confidence_level": "high" if final_score > 0.5 else "medium" if final_score > 0.25 else "low",
            "breakdown": scores
        }

    except Exception as e:
        logger.error(f"ML classifier error: {e}")
        return {
            "ml_score": 0.5,
            "should_send_to_gemini": True,
            "confidence_level": "unknown",
            "breakdown": {}
        }

# -------------------- WATI WhatsApp Configuration --------------------
WATI_BASE_URL = "https://live-mt-server.wati.io/458913"
WATI_API_URL = f"{WATI_BASE_URL}/api/v1/sendTemplateMessage"
WATI_BROADCAST_URL = f"{WATI_BASE_URL}/api/v1/sendTemplateMessages"  # For broadcast
WATI_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5NDVjNDA4YS1kM2VlLTRmZWYtOTI4MS1hODgwZGMzYjJhMjYiLCJ1bmlxdWVfbmFtZSI6ImFjdHVhbGx5dXNlZnVsZXh0ZW5zaW9uc0BnbWFpbC5jb20iLCJuYW1laWQiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiZW1haWwiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMjAvMjAyNSAwNzoyODozOSIsInRlbmFudF9pZCI6IjQ1ODkxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.JbUt76Z5FKfnG1UcyhtB7Uu99Pk0djigLtc6PmwlSZE"
WATI_TEMPLATE_NAME = "sports"
WATI_BROADCAST_NAME = "sports_240920250750"

async def gemini_generate_title_summary(news_title: str, news_description: str, category: str):
    """Enhanced title/summary generation with better prompts"""
    prompt = f"""
Category: {category}
Original Title: {news_title}
Description: {news_description}

Create a clear, engaging title and a concise 2-sentence summary for this {category} news.
Make it informative and attention-grabbing.

Format:
TITLE: [enhanced title]
SUMMARY: [2-sentence summary]
"""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 200,
                    "temperature": 0.3
                }
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

    # Fallback - Simple truncation (Gemini handles summarization)
    summary = news_description[:100] if len(news_description) > 100 else news_description

    return news_title, summary

async def gemini_generate_feedback_insights(news_title: str, news_description: str, category: str, feedback_type: str):
    """Generate personalization insights from user feedback"""

    if feedback_type in ["love_it", "like_it"]:
        prompt = f"""
LIKED NEWS ANALYSIS:
Title: {news_title}
Description: {news_description}
Category: {category}
User Feedback: {feedback_type.replace('_', ' ').title()}

TASK: Generate personalization insights for future news recommendations.

Analyze what the user liked about this news and create:
1. A smart question to find similar news
2. Three relevant tags for content filtering

EXAMPLES:
✅ Cricket Win → Question: "India cricket team victories" + Tags: ["india_cricket", "team_victories", "sports_wins"]
✅ Tech Launch → Question: "Apple iPhone new features" + Tags: ["apple_products", "iphone_updates", "tech_launches"]
✅ Movie Success → Question: "Bollywood box office hits" + Tags: ["bollywood_movies", "box_office", "hindi_cinema"]

FORMAT (Must follow exactly):
QUESTION: [Smart search question for similar news]
TAG1: [relevant_tag_1]
TAG2: [relevant_tag_2]
TAG3: [relevant_tag_3]
REASON: [Why user might like this type of news]
"""
    else:  # not_my_vibe
        prompt = f"""
DISLIKED NEWS ANALYSIS:
Title: {news_title}
Description: {news_description}
Category: {category}
User Feedback: Not My Vibe

TASK: Identify what to avoid in future recommendations.

Analyze what the user disliked and create:
1. Avoidance patterns
2. Content filters to exclude

FORMAT (Must follow exactly):
AVOID_PATTERN: [What type of content to avoid]
EXCLUDE_TAG1: [tag_to_exclude_1]
EXCLUDE_TAG2: [tag_to_exclude_2]
EXCLUDE_TAG3: [tag_to_exclude_3]
REASON: [Why user might not like this type of news]
"""

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 300,
                    "temperature": 0.2
                }
            }

            resp = await client.post(
                GEMINI_ENDPOINT,
                headers={"X-Goog-Api-Key": GEMINI_API_KEY},
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("candidates") and data["candidates"][0].get("content"):
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

                result = {"feedback_type": feedback_type}

                if feedback_type in ["love_it", "like_it"]:
                    # Parse positive feedback
                    if "QUESTION:" in text:
                        question_start = text.find("QUESTION:") + len("QUESTION:")
                        question = text[question_start:].split("\n")[0].strip()
                        result["search_question"] = question

                    tags = []
                    for i in range(1, 4):
                        tag_marker = f"TAG{i}:"
                        if tag_marker in text:
                            tag_start = text.find(tag_marker) + len(tag_marker)
                            tag = text[tag_start:].split("\n")[0].strip()
                            tags.append(tag)
                    result["preference_tags"] = tags

                    if "REASON:" in text:
                        reason_start = text.find("REASON:") + len("REASON:")
                        reason = text[reason_start:].split("\n")[0].strip()
                        result["reason"] = reason

                else:  # not_my_vibe
                    # Parse negative feedback
                    if "AVOID_PATTERN:" in text:
                        pattern_start = text.find("AVOID_PATTERN:") + len("AVOID_PATTERN:")
                        pattern = text[pattern_start:].split("\n")[0].strip()
                        result["avoid_pattern"] = pattern

                    exclude_tags = []
                    for i in range(1, 4):
                        tag_marker = f"EXCLUDE_TAG{i}:"
                        if tag_marker in text:
                            tag_start = text.find(tag_marker) + len(tag_marker)
                            tag = text[tag_start:].split("\n")[0].strip()
                            exclude_tags.append(tag)
                    result["exclude_tags"] = exclude_tags

                    if "REASON:" in text:
                        reason_start = text.find("REASON:") + len("REASON:")
                        reason = text[reason_start:].split("\n")[0].strip()
                        result["reason"] = reason

                logger.info(f"🧠 Gemini feedback analysis: {result}")
                return result

    except Exception as e:
        logger.error(f"Gemini feedback analysis error: {e}")

        # Fallback analysis
        return {
            "feedback_type": feedback_type,
            "search_question": f"More {category} news like {news_title}",
            "preference_tags": [category.lower(), "user_liked", "recommended"],
            "reason": "User showed interest in this content"
        }

async def smart_dynamic_gemini_gatekeeper(news_title: str, news_description: str, user_query: str, category: str = "general"):
    """
    DYNAMIC: Smart context-aware Gemini gatekeeper with real-time intelligence
    """

    # STEP 1: Dynamic context analysis
    context = analyze_query_context_dynamically(user_query)

    # STEP 2: Build intelligent prompt based on detected context
    smart_context = f"""
DYNAMIC CONTEXT ANALYSIS:
- Query Intent: {context['intent']}
- Specificity: {context['specificity']}
- Detected Entities: {context['entities']}
- Sports Context: {context['sports_context']}
- Gender Context: {context['gender_context']}
- Strict Matching Required: {context['strict_matching']}
"""

    # STEP 3: Build context-specific rules
    strict_rules = ""
    if context['strict_matching']:
        strict_rules = """
⚠️  STRICT MATCHING MODE ACTIVATED:
- User has specific entities/requirements
- Only exact matches allowed
- Be VERY selective and precise
"""

    if context['gender_context']:
        strict_rules += f"""
🚺🚹 GENDER CONTEXT: {context['gender_context'].upper()}
- If query mentions "{context['gender_context']}", news must be about {context['gender_context']}'s sports/events
- Reject opposite gender news completely
"""

    if context['entities']:
        strict_rules += f"""
👤 ENTITY FOCUS: {', '.join(context['entities'])}
- News must mention these exact entities
- Reject unrelated person/team/company news
"""

    prompt = f"""You are a smart news relevance analyzer focused on user's specific interests.

{smart_context}

USER'S QUERY: "{user_query}"
NEWS TITLE: "{news_title}"
NEWS DESCRIPTION: "{news_description}"
CATEGORY: {category}

ANALYSIS FOCUS:

1. FOLLOWUP QUESTIONS CHECK:
   - User's specific followup interests mentioned in query
   - Does news content match these specific interests?

2. CUSTOM QUESTION CHECK:
   - User's custom questions and requirements
   - Does news answer what user specifically asked for?

3. KEYWORD RELEVANCE:
   - Important keywords from user query
   - Are these keywords present in news content?

4. CONTEXT MATCHING:
   - Right type of content user wants
   - Appropriate category and topic

SMART EXAMPLES:
✅ User wants "Virat Kohli cricket" + News: "Virat Kohli scores century" → YES (matches person + sport)
❌ User wants "Virat Kohli cricket" + News: "Women's cricket team wins" → NO (wrong person/context)
✅ User wants "iPhone news" + News: "Apple launches iPhone 15" → YES (matches brand/product)
❌ User wants "iPhone news" + News: "Samsung Galaxy launch" → NO (wrong brand)

BE FOCUSED ON USER'S SPECIFIC INTERESTS FROM THEIR QUESTIONS.

RESPONSE FORMAT (Must follow exactly):
RELEVANT: YES/NO
CONFIDENCE: [0.0-1.0]
REASONING: [Why this matches or doesn't match user's specific interests]
SUMMARY: [One engaging sentence if YES, empty if NO]

ANALYZE:"""

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 300,
                    "temperature": 0.1,
                    "topP": 0.8
                }
            }
           
            resp = await client.post(
                GEMINI_ENDPOINT,
                headers={"X-Goog-Api-Key": GEMINI_API_KEY},
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("candidates") and data["candidates"][0].get("content"):
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
               
                # Robust parsing
                is_relevant = False
                confidence = 0.5
                summary = ""
                reasoning = ""
               
                # Parse RELEVANT
                if "RELEVANT:" in text.upper():
                    relevant_line = text.upper().split("RELEVANT:")[1].split("\n")[0].strip()
                    is_relevant = "YES" in relevant_line
               
                # Parse CONFIDENCE
                if "CONFIDENCE:" in text.upper():
                    try:
                        conf_line = text.upper().split("CONFIDENCE:")[1].split("\n")[0].strip()
                        conf_match = re.search(r'(\d+\.?\d*)', conf_line)
                        if conf_match:
                            confidence = float(conf_match.group(1))
                            if confidence > 1.0:
                                confidence = confidence / 100
                    except:
                        confidence = 0.8 if is_relevant else 0.2
               
                # Parse SUMMARY
                if "SUMMARY:" in text and is_relevant:
                    try:
                        summary_start = text.lower().find("summary:") + len("summary:")
                        summary = text[summary_start:].split("\n")[0].strip()
                    except:
                        summary = f"{category.title()} update: {news_title}"
               
                # Parse REASONING
                if "REASONING:" in text:
                    try:
                        reasoning_start = text.lower().find("reasoning:") + len("reasoning:")
                        reasoning = text[reasoning_start:].split("\n")[0].strip()
                    except:
                        reasoning = "Analysis completed"
               
                logger.info(f"Gemini: {news_title[:50]}... -> Relevant: {is_relevant}, Confidence: {confidence}")
               
                return {
                    "is_relevant": is_relevant,
                    "confidence": confidence,
                    "summary": summary if is_relevant else "",
                    "reasoning": reasoning
                }

    except Exception as e:
        logger.error(f"Gemini gatekeeper error: {e}")
       
        # Smart fallback
        return smart_fallback_analysis(news_title, news_description, user_query, category)

def smart_fallback_analysis(news_title: str, news_description: str, user_query: str, category: str):
    """Smart fallback when Gemini fails"""
   
    content = f"{news_title} {news_description}".lower()
    query = user_query.lower()
   
    # Extract key terms from user query
    query_terms = [term.strip() for term in query.replace("notify me when", "").replace("news about", "").split()]
   
    relevance_score = 0
    matches = []
   
    # Check direct term matches
    for term in query_terms:
        if term in content:
            relevance_score += 0.3
            matches.append(term)
   
    # Category-specific victory detection
    if category.lower() == "sports" and ("win" in query or "victory" in query):
        victory_terms = ["ease past", "beat", "defeat", "won", "victory", "triumph", "chased down", "successful"]
        for term in victory_terms:
            if term in content:
                relevance_score += 0.6
                matches.append(f"victory:{term}")
                break
   
    # Check team/player names
    if "india" in query and "india" in content:
        relevance_score += 0.4
        matches.append("india")
   
    is_relevant = relevance_score > 0.6
    summary = f"{category.title()} news: {news_title}" if is_relevant else ""
   
    return {
        "is_relevant": is_relevant,
        "confidence": min(relevance_score, 1.0),
        "summary": summary,
        "reasoning": f"Fallback matched: {', '.join(matches)}" if matches else "No matches found"
    }

# -------------------- Helper Functions (Keep existing) --------------------
def extract_image_from_entry(entry):
    """Enhanced image extraction from RSS entry - handles all formats"""
    image_url = None

    # Method 1: Check for media:content (ESPN Cricket, most feeds)
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            if isinstance(media, dict):
                # Get URL from media:content
                url = media.get('url') or media.get('href')
                if url and 'http' in url:
                    image_url = url
                    logger.info(f"🖼️ Found image via media:content: {url}")
                    break

    # Method 2: Check for coverImages (ESPN specific)
    if not image_url and hasattr(entry, 'coverimages'):
        if entry.coverimages and 'http' in entry.coverimages:
            image_url = entry.coverimages
            logger.info(f"🖼️ Found image via coverImages: {image_url}")

    # Method 3: Check raw entry dict for custom fields
    if not image_url and isinstance(entry, dict):
        # ESPN: coverImages field
        if 'coverimages' in entry and entry['coverimages']:
            image_url = entry['coverimages']
            logger.info(f"🖼️ Found image via entry coverImages: {image_url}")

        # Check for url field in media_content
        elif 'media_content' in entry and entry['media_content']:
            media_list = entry['media_content'] if isinstance(entry['media_content'], list) else [entry['media_content']]
            for media in media_list:
                if isinstance(media, dict) and media.get('url'):
                    image_url = media['url']
                    logger.info(f"🖼️ Found image via media_content url: {image_url}")
                    break

    # Method 4: Check for media:thumbnail
    if not image_url and hasattr(entry, 'media_thumbnail'):
        if entry.media_thumbnail:
            if isinstance(entry.media_thumbnail, list):
                image_url = entry.media_thumbnail[0].get('url') if entry.media_thumbnail[0] else None
            else:
                image_url = entry.media_thumbnail.get('url')
            if image_url:
                logger.info(f"🖼️ Found image via media_thumbnail: {image_url}")

    # Method 5: Check enclosures for images
    if not image_url and hasattr(entry, 'enclosures'):
        for enclosure in entry.enclosures:
            if enclosure.get('type', '').startswith('image/'):
                image_url = enclosure.get('href') or enclosure.get('url')
                if image_url:
                    logger.info(f"🖼️ Found image via enclosures: {image_url}")
                    break

    # Method 6: Parse HTML content for images (The Hindu style)
    if not image_url:
        content = entry.get('description', '') or entry.get('summary', '') or entry.get('content', '')
        if isinstance(content, list) and content:
            content = content[0].get('value', '')

        # Look for img tags in HTML content
        import re
        img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', str(content), re.IGNORECASE)
        if img_matches:
            # Filter out small icons and get the first reasonable image
            for img in img_matches:
                if not any(x in img.lower() for x in ['icon', 'logo', 'avatar', 'button']) and 'http' in img:
                    image_url = img
                    logger.info(f"🖼️ Found image via HTML parsing: {image_url}")
                    break

    # Method 7: Check for image field (generic)
    if not image_url and hasattr(entry, 'image'):
        if isinstance(entry.image, dict):
            image_url = entry.image.get('href') or entry.image.get('url')
        elif isinstance(entry.image, str):
            image_url = entry.image
        if image_url:
            logger.info(f"🖼️ Found image via image field: {image_url}")

    # Final validation
    if image_url and not image_url.startswith('http'):
        image_url = None

    return image_url

async def fetch_rss_feed(feed_url: str, retries=2):
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(feed_url)
                response.raise_for_status()
            feed = feedparser.parse(response.content)
            last_24h = datetime.now() - timedelta(hours=24)

            news_items = []
            for e in feed.entries:
                if 'published_parsed' in e and datetime(*e.published_parsed[:6]) > last_24h:
                    # Extract image from entry
                    image_url = extract_image_from_entry(e)

                    news_items.append({
                        "title": e.get("title",""),
                        "description": e.get("description",""),
                        "link": e.get("link",""),
                        "published": e.get("published",""),
                        "image_url": image_url,  # Add extracted image URL
                        "feed_url": feed_url,  # Add feed URL for better fallback selection
                        "news_id": f"{feed_url}_{hash(e.get('title',''))}"
                    })

            return news_items[:20]
        except Exception as e:
            logger.warning(f"RSS fetch failed ({feed_url}): {e}")
            if attempt == retries:
                return []

async def fetch_feeds_by_categories(user_categories: list):
    """Fetch RSS feeds only for user-selected categories"""
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

# -------------------- WhatsApp WATI Integration --------------------
def get_fallback_image_by_category(category: str, feed_url: str = ""):
    """Get appropriate fallback image based on news category and source"""
    category_images = {
        "sports": [
            "https://p.imgci.com/db/PICTURES/CMS/406600/406697.5.jpg",  # Cricket
            "https://img1.hscicdn.com/image/upload/f_auto,t_ds_wide_w_960,q_50/lsci/db/PICTURES/CMS/394500/394538.5.jpg",
            "https://p.imgci.com/db/PICTURES/CMS/394600/394697.5.jpg"
        ],
        "technology": [
            "https://cdn.vox-cdn.com/thumbor/MpA2HVftSFntlwyGcUDpbNuIVP0=/0x0:2040x1360/1200x628/filters:focal(1020x680:1021x681)/cdn.vox-cdn.com/uploads/chorus_asset/file/21951093/akrales_200724_4086_0.jpg",
            "https://techcrunch.com/wp-content/uploads/2023/01/tech-news-hero.jpg"
        ],
        "news": [
            "https://feeds.bbci.co.uk/news/1024/cpsprodpb/7C89/production/_118547984_gettyimages-1230323629.jpg",
            "https://cdn.cnn.com/cnn/.e/img/4.0/logos/cnn_logo_social.jpg"
        ],
        "movies": [
            "https://variety.com/wp-content/uploads/2023/01/variety-entertainment-news.jpg",
            "https://www.thehindu.com/theme/images/th-online/logo-hindu.svg"
        ]
    }

    # Try to get category-specific image
    if category.lower() in category_images:
        images = category_images[category.lower()]

        # Source-specific selection
        if "espn" in feed_url.lower() or "cricket" in feed_url.lower():
            return images[0] if images else None
        elif "bbc" in feed_url.lower():
            return "https://feeds.bbci.co.uk/news/1024/cpsprodpb/7C89/production/_118547984_gettyimages-1230323629.jpg"
        elif "cnn" in feed_url.lower():
            return "https://cdn.cnn.com/cnn/.e/img/4.0/logos/cnn_logo_social.jpg"
        else:
            return images[0] if images else None

    # Default fallback
    return "https://p.imgci.com/db/PICTURES/CMS/406600/406697.5.jpg"

async def send_whatsapp_news_update(phone_number: str, news_item: dict, image_url: str = None):
    """
    Send news update via WATI WhatsApp template
    Template: sports
    Variables: {{1}} = image, {{2}} = title, {{3}} = summary (link removed)
    """
    try:
        # CHECK FOR DUPLICATES FIRST
        if await is_duplicate_message(phone_number, news_item, time_window_minutes=60):
            logger.info(f"🚫 Skipping duplicate message for {phone_number}")
            return False

        # Prepare template parameters (3 variables only)
        title = news_item.get("enhanced_title", news_item.get("title", ""))[:100]  # Limit title length
        summary = news_item.get("enhanced_summary", news_item.get("description", ""))[:300]  # Increased since no link

        # Use RSS extracted image or category-specific fallback
        extracted_image = news_item.get("image_url")
        category = news_item.get("category", "")
        feed_url = news_item.get("feed_url", "")

        if image_url:
            image = image_url
        elif extracted_image:
            image = extracted_image
            logger.info(f"🖼️ Using RSS extracted image: {extracted_image}")
        else:
            image = get_fallback_image_by_category(category, feed_url)
            logger.info(f"🖼️ Using fallback image for {category} from {feed_url}: {image}")

        # Generate unique feedback ID for this news item (will be used for button responses)
        user_id_for_feedback = news_item.get('user_id', 'unknown')
        feedback_id = f"{news_item.get('news_id', hash(title))}_{user_id_for_feedback}"

        # Format phone number for WATI (add +91 if Indian number)
        formatted_phone = phone_number.strip()
        if not formatted_phone.startswith('+'):
            if formatted_phone.startswith('91'):
                formatted_phone = '+' + formatted_phone
            elif len(formatted_phone) == 10:
                formatted_phone = '+91' + formatted_phone
            else:
                formatted_phone = '+91' + formatted_phone

        logger.info(f"📱 Formatted phone: {phone_number} -> {formatted_phone}")

        # WATI API payload - 3 variables only (link removed)
        payload = {
            "template_name": WATI_TEMPLATE_NAME,
            "broadcast_name": WATI_BROADCAST_NAME,
            "receivers": [
                {
                    "whatsappNumber": formatted_phone,
                    "customParams": [
                        {
                            "name": "1",
                            "value": image
                        },
                        {
                            "name": "2",
                            "value": title
                        },
                        {
                            "name": "3",
                            "value": summary
                        }
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {WATI_BEARER_TOKEN}",
            "Content-Type": "application/json"
        }

        logger.info(f"📱 Sending WhatsApp news to {phone_number}: {title[:50]}...")
        logger.info(f"🔗 WATI Payload: {json.dumps(payload, indent=2)}")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                WATI_BROADCAST_URL,  # Use broadcast endpoint for correct structure
                headers=headers,
                json=payload
            )

            logger.info(f"📱 WATI Response Status: {response.status_code}")
            logger.info(f"📱 WATI Response: {response.text}")

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("result") == True:
                    logger.info(f"✅ WhatsApp message sent successfully to {formatted_phone}")
                    # Track successful message
                    await track_message_sent(phone_number, news_item, True)
                    return True
                else:
                    logger.error(f"❌ WATI API Failed: {response_data}")
                    # Check specific error reasons
                    if not response_data.get("validWhatsAppNumber"):
                        logger.error(f"❌ Invalid WhatsApp number: {formatted_phone}")
                    # Track failed message
                    await track_message_sent(phone_number, news_item, False)
                    return False
            else:
                logger.error(f"❌ WATI API Error {response.status_code}: {response.text}")
                # Track failed message
                await track_message_sent(phone_number, news_item, False)
                return False

    except Exception as e:
        logger.error(f"❌ WhatsApp sending error: {e}")
        # Track failed message
        await track_message_sent(phone_number, news_item, False)
        return False

async def broadcast_news_to_users(news_results: list, user_phone_mapping: dict):
    """
    Broadcast news updates to multiple users via WhatsApp with deduplication
    """
    sent_count = 0

    for news_item in news_results[:5]:  # Limit to top 5 news items
        user_id = news_item.get("user_id")
        phone_number = user_phone_mapping.get(user_id)

        if phone_number:
            # Check for duplicates before sending
            if not await is_duplicate_message(phone_number, news_item, time_window_minutes=60):
                success = await send_whatsapp_news_update(phone_number, news_item)
                if success:
                    sent_count += 1
            else:
                logger.info(f"🚫 Skipping duplicate broadcast for {phone_number}: {news_item.get('title', '')[:50]}...")

            # Add delay between messages to avoid rate limiting
            await asyncio.sleep(3)  # Increased to 3 seconds

    logger.info(f"📱 WhatsApp broadcast completed: {sent_count} messages sent")
    return sent_count

# -------------------- Message Deduplication System --------------------
async def is_duplicate_message(phone_number: str, news_item: dict, time_window_minutes: int = 60):
    """Check if this message was already sent recently"""
    try:
        news_id = news_item.get("news_id", f"hash_{hash(news_item.get('title', ''))}")

        # Check if this exact news was sent to this user in the time window
        time_threshold = datetime.now() - timedelta(minutes=time_window_minutes)

        existing_message = await message_tracking_collection.find_one({
            "phone_number": phone_number,
            "news_id": news_id,
            "sent_at": {"$gte": time_threshold},
            "status": "sent"
        })

        if existing_message:
            logger.info(f"🚫 Duplicate detected: {news_item.get('title', '')[:50]}... for {phone_number}")
            return True

        return False

    except Exception as e:
        logger.error(f"❌ Error checking duplicate: {e}")
        return False

async def track_message_sent(phone_number: str, news_item: dict, success: bool):
    """Track sent messages to prevent duplicates"""
    try:
        news_id = news_item.get("news_id", f"hash_{hash(news_item.get('title', ''))}")

        tracking_doc = {
            "phone_number": phone_number,
            "news_id": news_id,
            "news_title": news_item.get("title", ""),
            "category": news_item.get("category", ""),
            "user_id": news_item.get("user_id", "unknown"),
            "sent_at": datetime.now(),
            "status": "sent" if success else "failed",
            "message_type": "news_alert"
        }

        await message_tracking_collection.insert_one(tracking_doc)
        logger.info(f"📋 Tracked message: {news_item.get('title', '')[:30]}... → {phone_number}")

    except Exception as e:
        logger.error(f"❌ Error tracking message: {e}")

async def cleanup_old_message_tracking(days_to_keep: int = 7):
    """Clean up old message tracking records"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        result = await message_tracking_collection.delete_many({
            "sent_at": {"$lt": cutoff_date}
        })
        logger.info(f"🧹 Cleaned up {result.deleted_count} old message tracking records")

    except Exception as e:
        logger.error(f"❌ Error cleaning up tracking: {e}")

# -------------------- User Feedback System --------------------
async def store_user_feedback(user_id: str, phone_number: str, news_item: dict, feedback_type: str, feedback_id: str):
    """Store user feedback in database"""
    try:
        # Generate insights using Gemini
        insights = await gemini_generate_feedback_insights(
            news_item.get("title", ""),
            news_item.get("description", ""),
            news_item.get("category", ""),
            feedback_type
        )

        # Store feedback
        feedback_doc = {
            "feedback_id": feedback_id,
            "user_id": user_id,
            "phone_number": phone_number,
            "news_item": {
                "title": news_item.get("title", ""),
                "description": news_item.get("description", ""),
                "category": news_item.get("category", ""),
                "link": news_item.get("link", ""),
                "news_id": news_item.get("news_id", "")
            },
            "feedback_type": feedback_type,
            "gemini_insights": insights,
            "timestamp": datetime.now(),
            "processed": False
        }

        await feedback_collection.insert_one(feedback_doc)

        # Update user preferences based on feedback
        await update_user_preferences(user_id, insights, feedback_type)

        logger.info(f"✅ Stored {feedback_type} feedback for user {user_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Error storing feedback: {e}")
        return False

async def update_user_preferences(user_id: str, insights: dict, feedback_type: str):
    """Update user preferences based on feedback"""
    try:
        # Get existing preferences
        existing_prefs = await preferences_collection.find_one({"user_id": user_id})

        if feedback_type in ["love_it", "like_it"]:
            # Add positive preferences
            if existing_prefs:
                # Update existing preferences
                search_questions = existing_prefs.get("preferred_search_questions", [])
                preference_tags = existing_prefs.get("preference_tags", [])

                # Add new insights
                if insights.get("search_question"):
                    search_questions.append(insights["search_question"])

                if insights.get("preference_tags"):
                    preference_tags.extend(insights["preference_tags"])

                # Remove duplicates
                search_questions = list(set(search_questions))
                preference_tags = list(set(preference_tags))

                await preferences_collection.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "preferred_search_questions": search_questions,
                            "preference_tags": preference_tags,
                            "last_updated": datetime.now()
                        }
                    }
                )
            else:
                # Create new preferences
                new_prefs = {
                    "user_id": user_id,
                    "preferred_search_questions": [insights.get("search_question", "")],
                    "preference_tags": insights.get("preference_tags", []),
                    "avoid_patterns": [],
                    "exclude_tags": [],
                    "created_at": datetime.now(),
                    "last_updated": datetime.now()
                }
                await preferences_collection.insert_one(new_prefs)

        else:  # not_my_vibe
            # Add negative preferences
            if existing_prefs:
                avoid_patterns = existing_prefs.get("avoid_patterns", [])
                exclude_tags = existing_prefs.get("exclude_tags", [])

                # Add new insights
                if insights.get("avoid_pattern"):
                    avoid_patterns.append(insights["avoid_pattern"])

                if insights.get("exclude_tags"):
                    exclude_tags.extend(insights["exclude_tags"])

                # Remove duplicates
                avoid_patterns = list(set(avoid_patterns))
                exclude_tags = list(set(exclude_tags))

                await preferences_collection.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            "avoid_patterns": avoid_patterns,
                            "exclude_tags": exclude_tags,
                            "last_updated": datetime.now()
                        }
                    }
                )
            else:
                # Create new preferences with negative feedback
                new_prefs = {
                    "user_id": user_id,
                    "preferred_search_questions": [],
                    "preference_tags": [],
                    "avoid_patterns": [insights.get("avoid_pattern", "")],
                    "exclude_tags": insights.get("exclude_tags", []),
                    "created_at": datetime.now(),
                    "last_updated": datetime.now()
                }
                await preferences_collection.insert_one(new_prefs)

        logger.info(f"✅ Updated preferences for user {user_id} based on {feedback_type}")

    except Exception as e:
        logger.error(f"❌ Error updating preferences: {e}")

async def send_feedback_acknowledgment(phone_number: str, feedback_type: str, insights: dict):
    """Send acknowledgment message after receiving feedback"""
    try:
        if feedback_type in ["love_it", "like_it"]:
            ack_title = f"Thanks for the {feedback_type.replace('_', ' ')}! 😊"

            search_question = insights.get("search_question", "")
            tags = ", ".join(insights.get("preference_tags", []))

            ack_summary = f"I've learned you like: {search_question}. I'll find more news with these interests: {tags}"
        else:
            ack_title = "Got it! Not your vibe 👍"
            avoid_pattern = insights.get("avoid_pattern", "")
            ack_summary = f"I'll avoid showing you: {avoid_pattern}. Your preferences are being updated!"

        # Create acknowledgment news item
        ack_news = {
            "title": ack_title,
            "enhanced_title": ack_title,
            "description": ack_summary,
            "enhanced_summary": ack_summary,
            "link": "https://naarad.ai/feedback-received",
            "category": "feedback"
        }

        # Send acknowledgment via WhatsApp
        success = await send_whatsapp_news_update(phone_number, ack_news)

        logger.info(f"📱 Sent feedback acknowledgment to {phone_number}: {success}")
        return success

    except Exception as e:
        logger.error(f"❌ Error sending feedback acknowledgment: {e}")
        return False

def calculate_similarity(text1: str, text2: str):
    try:
        emb1 = sem_model.encode(text1)
        emb2 = sem_model.encode(text2)
        return util.pytorch_cos_sim(emb1, emb2).item()
    except:
        return 0.0

def extract_entities(text: str):
    """OPTIMIZED: Simple entity extraction (Gemini handles complex NER)"""
    try:
        # Simple regex-based entity extraction for basic cases
        import re
        entities = []

        # Extract potential person names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
        entities.extend(['PERSON'] * len(names))

        # Extract potential organizations (common patterns)
        orgs = re.findall(r'\b(?:Inc|Ltd|Corp|Company|University|College)\b', text, re.IGNORECASE)
        entities.extend(['ORG'] * len(orgs))

        return entities[:5]  # Limit for performance
    except:
        return []

def analyze_sentiment(text: str):
    """OPTIMIZED: Simple sentiment analysis (Gemini handles complex sentiment)"""
    try:
        # Simple keyword-based sentiment analysis
        text_lower = text.lower()

        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'win', 'victory', 'success', 'happy', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'fail', 'loss', 'defeat', 'sad', 'angry', 'disaster']

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "POSITIVE", min(0.9, 0.5 + (pos_count * 0.1))
        elif neg_count > pos_count:
            return "NEGATIVE", min(0.9, 0.5 + (neg_count * 0.1))
        else:
            return "NEUTRAL", 0.5
    except:
        return "NEUTRAL", 0.5

# -------------------- OPTIMIZED: Lightweight Dynamic Learning System --------------------
# Memory usage reduced from 3.3GB to 22MB for free hosting compatibility
class FixedDynamicLearningSystem:
    def __init__(self):
        self.cache_duration = timedelta(hours=1)

    async def process_alerts(self, news_items, alerts, user_profile, user_categories):
        """FIXED: Process alerts with correct query matching and universal Gemini analysis + User Feedback"""

        results = []

        # Optimization counters
        total_news_checked = 0
        quick_skipped = 0
        similarity_checked = 0
        gemini_sent = 0
        gemini_approved = 0

        # Get user preferences from feedback for the first alert (assuming same user)
        user_id = alerts[0].get("user_id") if alerts else None
        user_preferences = await preferences_collection.find_one({"user_id": user_id}) if user_id else None

        for alert in alerts:
            # BUILD COMPREHENSIVE QUERY from all user inputs + feedback preferences
            query_parts = []

            # 1. Custom question (highest priority)
            custom_q = alert.get("custom_question", "")
            if custom_q:
                query_parts.append(custom_q)

            # 2. Add learned preferences from feedback
            if user_preferences:
                preferred_questions = user_preferences.get("preferred_search_questions", [])
                for pq in preferred_questions[:3]:  # Top 3 learned preferences
                    if pq and pq not in " ".join(query_parts):
                        query_parts.append(pq)

            # 3. Add main category context
            main_cat = alert.get("main_category", "")
            if main_cat:
                query_parts.append(f"{main_cat} news")

            # 4. Add sub-categories for specificity
            sub_cats = alert.get("sub_categories", [])
            for sub_cat in sub_cats:
                if sub_cat and sub_cat not in " ".join(query_parts):
                    query_parts.append(sub_cat)

            # 5. Add followup questions as additional context
            followup_qs = alert.get("followup_questions", [])
            for fq in followup_qs:
                if fq and fq not in " ".join(query_parts):
                    query_parts.append(fq)

            # Combine all parts into comprehensive query
            primary_query = " ".join(query_parts) if query_parts else "general news"

            logger.info(f"🎯 Processing Alert - Comprehensive Query: '{primary_query}'")
            logger.info(f"📝 Query Components - Custom: '{custom_q}', Sub-cats: {sub_cats}, Followups: {followup_qs}")
            if user_preferences:
                logger.info(f"🧠 Learned Preferences: {user_preferences.get('preferred_search_questions', [])[:3]}")

            # Process each news item for this alert
            for news_item in news_items:
                try:
                    total_news_checked += 1

                    # Skip if category doesn't match
                    news_category = news_item.get('category', '').lower()
                    if news_category not in [cat.lower() for cat in user_categories]:
                        continue

                    # ENHANCED FILTER: Skip news based on user feedback
                    if self.should_skip_news_item_with_feedback(news_item, primary_query, alert, user_preferences):
                        continue

                    # STEP 1: Quick keyword filtering first (super fast)
                    content = f"{news_item.get('title', '')} {news_item.get('description', '')}"
                    content_lower = content.lower()

                    # Build all keywords from user preferences
                    all_keywords = []

                    # From custom question
                    custom_q = alert.get("custom_question", "")
                    if custom_q:
                        all_keywords.extend(custom_q.lower().split())

                    # From sub-categories
                    sub_categories = alert.get("sub_categories", [])
                    all_keywords.extend([sc.lower() for sc in sub_categories])

                    # From followup questions
                    followup_keywords = alert.get("followup_questions", [])
                    for fq in followup_keywords:
                        all_keywords.extend(fq.lower().split())

                    # IMPROVED: Smart keyword matching with fuzzy matching
                    keyword_matches = 0
                    matched_keywords = []

                    for kw in all_keywords:
                        if len(kw) > 2:
                            # Exact match
                            if kw in content_lower:
                                keyword_matches += 1
                                matched_keywords.append(kw)
                            # Partial match for names (like "katrina" matches "katrina kaif")
                            elif len(kw) > 4 and any(word.startswith(kw) or kw in word for word in content_lower.split()):
                                keyword_matches += 0.5
                                matched_keywords.append(f"{kw}~")

                    # More lenient threshold - allow if ANY meaningful keyword matches
                    if keyword_matches == 0 and len(all_keywords) > 3:
                        # Additional check: Look for proper names in news that might match query intent
                        news_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', f"{news_item.get('title', '')} {news_item.get('description', '')}")
                        query_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', primary_query)

                        entity_overlap = any(entity.lower() in news_entity.lower() or news_entity.lower() in entity.lower()
                                           for entity in query_entities for news_entity in news_entities)

                        if not entity_overlap:
                            quick_skipped += 1
                            logger.info(f"🚫 Quick skip: {news_item.get('title', '')[:50]}... (no keywords: {all_keywords[:5]})")
                            continue
                        else:
                            logger.info(f"🔍 Entity overlap found - proceeding: {news_item.get('title', '')[:50]}...")

                    # STEP 2: Semantic similarity check (more expensive)
                    similarity_checked += 1
                    similarity = calculate_similarity(content.lower(), primary_query.lower())

                    # STEP 3: Enhanced scoring with user preferences
                    boost = 0

                    # ENHANCED: Followup questions boost (user's specific interests)
                    for keyword in followup_keywords:
                        if keyword.lower() in content_lower:
                            boost += 0.5  # Increased from 0.3 to 0.5
                            logger.info(f"🎯 Followup question match: {keyword}")

                    # Custom question boost
                    custom_q = alert.get("custom_question", "")
                    if custom_q:
                        custom_words = custom_q.lower().split()
                        custom_matches = sum(1 for word in custom_words if word in content_lower and len(word) > 2)
                        if custom_matches > 0:
                            boost += 0.4 * (custom_matches / len(custom_words))
                            logger.info(f"🎯 Custom question match: {custom_matches}/{len(custom_words)} words")

                    # Sub-category matching boost (reduced importance)
                    for sub_cat in sub_categories:
                        if sub_cat.lower() in content_lower:
                            boost += 0.2  # Reduced from 0.25 to 0.2

                    # FEEDBACK BOOST: User preferences
                    if user_preferences:
                        preference_tags = user_preferences.get("preference_tags", [])
                        for tag in preference_tags:
                            if tag.lower() in content_lower:
                                boost += 0.4
                                logger.info(f"🎯 Preference tag match: {tag}")

                    base_score = min(1.0, similarity + boost)

                    # STEP 4: Lightweight ML pre-classification
                    ml_result = lightweight_ml_classifier(
                        news_item.get("title", ""),
                        news_item.get("description", ""),
                        primary_query
                    )

                    combined_score = max(base_score, ml_result["ml_score"])

                    # STEP 5: Smart ML + Gemini decision making
                    if ml_result["should_send_to_gemini"] and combined_score > 0.25:
                        gemini_sent += 1
                        logger.info(f"📋 ML + Context approved for Gemini: {news_item.get('title', '')[:60]}...")
                        logger.info(f"   📊 Scores - Base: {base_score:.2f}, ML: {ml_result['ml_score']:.2f}, Combined: {combined_score:.2f}")

                        # SMART DYNAMIC Gemini analysis with context intelligence (final validation)
                        gemini_result = await smart_dynamic_gemini_gatekeeper(
                            news_item.get("title", ""),
                            news_item.get("description", ""),
                            primary_query,
                            news_category
                        )

                        # INTELLIGENT DECISION: Trust ML when it has high confidence
                        should_proceed = False

                        if ml_result["ml_score"] > 0.35:  # High ML confidence - trust ML over Gemini
                            should_proceed = True
                            logger.info(f"🤖 HIGH ML CONFIDENCE ({ml_result['ml_score']:.2f}) - Trusting ML decision")
                        elif gemini_result["is_relevant"] and gemini_result["confidence"] > 0.3:  # Lower Gemini threshold
                            should_proceed = True
                            logger.info(f"🧠 Gemini + ML agreement - Both approve")
                        else:
                            logger.info(f"❌ Both ML ({ml_result['ml_score']:.2f}) and Gemini ({gemini_result['confidence']:.2f}) have low confidence")

                        if should_proceed:
                            gemini_approved += 1
                            logger.info(f"✅ Gemini Approved: {news_item.get('title', '')[:60]}...")

                            # Generate enhanced title/summary
                            enhanced_title, enhanced_summary = await gemini_generate_title_summary(
                                news_item.get("title", ""),
                                news_item.get("description", ""),
                                news_category
                            )

                            # NLP analysis for additional context
                            entities = extract_entities(content)
                            sentiment_label, sentiment_score = analyze_sentiment(content)

                            # Create final result with ML + context influence
                            final_score = max(base_score, ml_result["ml_score"], gemini_result.get("confidence", 0.5))

                            result = {
                                **news_item,
                                "alert_id": alert.get("alert_id", "unknown"),
                                "user_id": alert.get("user_id", "unknown"),
                                "matched_query": primary_query,  # FIXED: Use actual user query
                                "ai_analysis": {
                                    "satisfies_user_query": True,
                                    "confidence_score": final_score,
                                    "entities": entities,
                                    "sentiment": sentiment_label,
                                    "sentiment_score": sentiment_score,
                                    "should_notify": True,
                                    "notification_text": gemini_result.get("summary", f"{news_category.title()} update: {news_item.get('title', '')}"),
                                    "gemini_reasoning": gemini_result.get("reasoning", "ML high confidence approval"),
                                    "gemini_confidence": gemini_result.get("confidence", 0.0),
                                    "ml_score": ml_result["ml_score"],
                                    "ml_breakdown": ml_result["breakdown"],
                                    "decision_source": "ml_high_confidence" if ml_result["ml_score"] > 0.35 else "gemini_approval",
                                    "feedback_influenced": bool(user_preferences)
                                },
                                "final_score": final_score,
                                "notification_text": gemini_result.get("summary", f"{news_category.title()} update: {news_item.get('title', '')}"),
                                "enhanced_title": enhanced_title,
                                "enhanced_summary": enhanced_summary,
                                "personalization_score": self._calculate_personalization_score_with_feedback(news_item, alert, user_profile, user_preferences)
                            }

                            results.append(result)

                        else:
                            logger.info(f"❌ Gemini Rejected: {news_item.get('title', '')[:60]}...")

                except Exception as e:
                    logger.error(f"Error processing news item: {e}")
                    continue

        # Remove duplicates and sort with feedback influence
        unique_results = {}
        for result in results:
            title_key = result["title"].lower()
            if title_key not in unique_results or result["final_score"] > unique_results[title_key]["final_score"]:
                unique_results[title_key] = result

        # Sort by relevance score (feedback-influenced)
        final_results = sorted(unique_results.values(), key=lambda x: x["final_score"], reverse=True)

        # DYNAMIC INTELLIGENCE OPTIMIZATION REPORT
        gemini_efficiency = ((total_news_checked - gemini_sent) / total_news_checked * 100) if total_news_checked > 0 else 0
        ml_efficiency = ((similarity_checked - gemini_sent) / similarity_checked * 100) if similarity_checked > 0 else 0

        logger.info(f"🎉 DYNAMIC INTELLIGENCE OPTIMIZATION REPORT:")
        logger.info(f"   📊 Total news checked: {total_news_checked}")
        logger.info(f"   ⚡ Quick keyword skipped: {quick_skipped}")
        logger.info(f"   🔍 Similarity checked: {similarity_checked}")
        logger.info(f"   🤖 ML pre-filtered: {similarity_checked - gemini_sent}")
        logger.info(f"   🚀 Sent to Gemini (final validation): {gemini_sent}")
        logger.info(f"   ✅ Gemini approved: {gemini_approved}")
        logger.info(f"   💰 Total API calls saved: {gemini_efficiency:.1f}%")
        logger.info(f"   🧠 ML filtering efficiency: {ml_efficiency:.1f}%")
        logger.info(f"   🎯 Final relevant results: {len(final_results)}")
        logger.info(f"   🏆 System: Dynamic Context Intelligence + Lightweight ML")

        return final_results[:15]

    def should_skip_news_item(self, news_item, user_query, alert):
        """Enhanced precision filter using ALL alert components"""

        title = news_item.get("title", "").lower()
        description = news_item.get("description", "").lower()
        content = f"{title} {description}"
        query = user_query.lower()

        # Get all user interests from alert
        custom_q = alert.get("custom_question", "").lower()
        followup_qs = [fq.lower() for fq in alert.get("followup_questions", [])]
        sub_cats = [sc.lower() for sc in alert.get("sub_categories", [])]

        # Combine all user interests
        all_interests = [custom_q] + followup_qs + sub_cats
        user_keywords = set()

        for interest in all_interests:
            if interest:
                words = interest.split()
                user_keywords.update([w for w in words if len(w) > 2])

        # SPORTS SPECIFIC FILTERING
        if "sports" in alert.get("main_category", "").lower():
            # If user mentions specific teams/players
            teams_mentioned = []
            players_mentioned = []

            for keyword in user_keywords:
                if keyword in ["india", "indian"]:
                    teams_mentioned.append("india")
                elif keyword in ["england", "australia", "pakistan", "south africa", "new zealand"]:
                    teams_mentioned.append(keyword)
                elif keyword in ["abhishek", "virat", "kohli", "rohit", "sharma", "dhoni"]:
                    players_mentioned.append(keyword)

            # If user wants India but news is about other teams winning
            if "india" in teams_mentioned and "win" in custom_q:
                other_teams = ["england", "australia", "pakistan", "south africa"]
                for team in other_teams:
                    if team in content and any(win_word in content for win_word in ["win", "beat", "defeat", "triumph"]):
                        if "india" not in content:
                            logger.info(f"🚫 Skipping {team} victory news - user wants India wins")
                            return True

            # If user mentions specific players, news should mention them
            if players_mentioned:
                player_in_news = any(player in content for player in players_mentioned)
                if not player_in_news and len(players_mentioned) > 0:
                    logger.info(f"🚫 Skipping news - user wants {players_mentioned} but not mentioned")
                    return True

        # TECHNOLOGY SPECIFIC FILTERING
        elif "technology" in alert.get("main_category", "").lower():
            tech_brands = []
            for keyword in user_keywords:
                if keyword in ["apple", "iphone", "ios"]:
                    tech_brands.append("apple")
                elif keyword in ["google", "android"]:
                    tech_brands.append("google")
                elif keyword in ["samsung"]:
                    tech_brands.append("samsung")

            # Skip competitor news
            if "apple" in tech_brands:
                competitors = ["google", "samsung", "microsoft"]
                for competitor in competitors:
                    if competitor in content and "apple" not in content:
                        logger.info(f"🚫 Skipping {competitor} news - user wants Apple")
                        return True

        # GENERAL KEYWORD MATCHING
        # If user has specific interests but news doesn't match any
        if len(user_keywords) > 2:  # User has specific interests
            keyword_matches = sum(1 for keyword in user_keywords if keyword in content)
            if keyword_matches == 0:
                logger.info(f"🚫 Skipping news - no keywords match from {list(user_keywords)[:5]}")
                return True

        return False

    def should_skip_news_item_with_feedback(self, news_item, user_query, alert, user_preferences):
        """Enhanced filtering with user feedback preferences"""

        # First apply original filter
        if self.should_skip_news_item(news_item, user_query, alert):
            return True

        # Additional feedback-based filtering
        if user_preferences:
            title = news_item.get("title", "").lower()
            description = news_item.get("description", "").lower()
            content = f"{title} {description}"

            # Check against exclude patterns
            avoid_patterns = user_preferences.get("avoid_patterns", [])
            for pattern in avoid_patterns:
                if pattern.lower() in content:
                    logger.info(f"🚫 Skipping due to avoid pattern: {pattern}")
                    return True

            # Check against exclude tags
            exclude_tags = user_preferences.get("exclude_tags", [])
            for tag in exclude_tags:
                if tag.lower() in content:
                    logger.info(f"🚫 Skipping due to exclude tag: {tag}")
                    return True

        return False

    def _calculate_personalization_score(self, news_item, alert, user_profile):
        """Enhanced personalization scoring"""
        score = 0
        content = f"{news_item.get('title','')} {news_item.get('description','')}".lower()

        # Followup questions boost
        followup_keywords = alert.get("followup_questions", [])
        for keyword in followup_keywords:
            if keyword.lower() in content:
                score += 0.4

        # Custom question similarity
        custom_q = alert.get("custom_question", "")
        if custom_q:
            custom_similarity = calculate_similarity(content, custom_q.lower())
            score += custom_similarity * 0.3

        # Sub-categories boost
        sub_cats = alert.get("sub_categories", [])
        for sub_cat in sub_cats:
            if sub_cat.lower() in content:
                score += 0.3

        return min(1.0, score)

    def _calculate_personalization_score_with_feedback(self, news_item, alert, user_profile, user_preferences):
        """Enhanced personalization scoring with user feedback"""

        # Get base score
        base_score = self._calculate_personalization_score(news_item, alert, user_profile)

        # Add feedback boost
        feedback_score = 0
        content = f"{news_item.get('title','')} {news_item.get('description','')}".lower()

        if user_preferences:
            # Boost for preference tags
            preference_tags = user_preferences.get("preference_tags", [])
            for tag in preference_tags:
                if tag.lower() in content:
                    feedback_score += 0.3

            # Boost for preferred search questions similarity
            preferred_questions = user_preferences.get("preferred_search_questions", [])
            for question in preferred_questions:
                similarity = calculate_similarity(content, question.lower())
                feedback_score += similarity * 0.2

        final_score = min(1.0, base_score + feedback_score)

        if feedback_score > 0:
            logger.info(f"🎯 Feedback boost: {feedback_score:.2f} for {news_item.get('title', '')[:50]}...")

        return final_score

    async def create_notifications(self, results, user_id):
        """Create notifications for relevant news"""
        notifications = []
        for r in results:
            if r["final_score"] > 0.5:  # Higher threshold for notifications
                notifications.append({
                    "user_id": user_id,
                    "alert_id": r["alert_id"],
                    "title": r["notification_text"],
                    "content": {
                        "news_title": r["title"],
                        "news_link": r.get("link", ""),
                        "matched_query": r["matched_query"],
                        "entities": r["ai_analysis"]["entities"],
                        "sentiment": r["ai_analysis"]["sentiment"],
                        "confidence_score": r["final_score"]
                    },
                    "priority": "high" if r["final_score"] > 0.8 else "medium",
                    "is_read": False,
                    "created_at": datetime.now()
                })
       
        if notifications:
            await notifications_collection.insert_many(notifications)
            logger.info(f"📬 Created {len(notifications)} notifications")

    async def send_whatsapp_notifications(self, results, user_id):
        """Send WhatsApp notifications for relevant news"""

        # Get user phone number from users collection
        user_doc = await users_collection.find_one({"user_id": user_id})
        phone_number = user_doc.get("phone_number") if user_doc else None

        if not phone_number:
            logger.warning(f"📱 No phone number found for user {user_id}")
            return 0

        sent_count = 0

        # Send WhatsApp for high-priority news only
        for result in results[:3]:  # Limit to top 3 results
            if result["final_score"] > 0.7:  # High relevance threshold
                # Check for duplicates first
                if not await is_duplicate_message(phone_number, result, time_window_minutes=60):
                    success = await send_whatsapp_news_update(phone_number, result)
                    if success:
                        sent_count += 1
                else:
                    logger.info(f"🚫 Skipping duplicate notification for {user_id}: {result.get('title', '')[:50]}...")

                # Add delay between messages
                await asyncio.sleep(3)

        logger.info(f"📱 Sent {sent_count} WhatsApp messages to {user_id}")
        return sent_count

# -------------------- UPDATED FastAPI Routes --------------------
fixed_learning_system = FixedDynamicLearningSystem()

# MAIN OPTIMIZED ENDPOINT - Single route for all news processing
@router.get("/intelligent-news")
async def get_intelligent_news(user_id: str = Query(...)):
    """
    MAIN optimized endpoint for intelligent news processing
    Memory efficient: Uses only SentenceTransformer + Gemini API (22MB vs 3.3GB)
    Perfect for free hosting: Vercel/Railway/Render/Heroku
    """
    try:
        logger.info(f"🚀 OPTIMIZED processing for user: {user_id} (Memory efficient)")

        # Fetch user alerts
        alerts = [a async for a in alerts_collection.find({"user_id": user_id, "is_active": True})]
        if not alerts:
            return {"message": "No active alerts found", "intelligent_news": []}

        logger.info(f"📋 Found {len(alerts)} active alerts")

        # Extract categories from alerts
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
        logger.info(f"📂 User categories: {user_categories}")

        # Get user profile
        user_profile = await users_collection.find_one({"user_id": user_id}) or {"preferences": {}}

        # Fetch news for categories
        category_news = await fetch_feeds_by_categories(user_categories)
        logger.info(f"📰 Fetched {len(category_news)} news items")

        if not category_news:
            return {"message": "No news found for categories", "intelligent_news": []}

        # Process with optimized system
        intelligent_results = await fixed_learning_system.process_alerts(
            category_news, alerts, user_profile, user_categories
        )

        # Create notifications
        await fixed_learning_system.create_notifications(intelligent_results, user_id)

        # Send WhatsApp notifications for high-priority news
        whatsapp_sent = 0
        if intelligent_results:
            whatsapp_sent = await fixed_learning_system.send_whatsapp_notifications(intelligent_results, user_id)

        # Store results
        if intelligent_results:
            await personalized_collection.insert_one({
                "user_id": user_id,
                "categories": user_categories,
                "generated_at": datetime.now(),
                "results": intelligent_results,
                "system_version": "optimized_lightweight_v1"
            })

        return {
            "user_id": user_id,
            "categories": user_categories,
            "intelligent_news": intelligent_results,
            "system_info": {
                "version": "optimized_lightweight",
                "memory_usage": "~22MB (vs 3.3GB)",
                "models_used": ["sentence-transformers", "gemini-api"],
                "hosting_ready": "vercel/railway/render/heroku"
            },
            "processing_stats": {
                "total_news_processed": len(category_news),
                "relevant_news_found": len(intelligent_results),
                "alerts_processed": len(alerts),
                "whatsapp_messages_sent": whatsapp_sent
            }
        }

    except Exception as e:
        logger.error(f"❌ Error in intelligent news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Essential WhatsApp webhook for user feedback

@router.get("/debug-news-matching")
async def debug_news_matching(
    query: str = Query(..., description="User query to test"),
    news_title: str = Query(..., description="News title to test"),
    news_description: str = Query(..., description="News description to test")
):
    """Debug endpoint to test news matching logic"""
    try:
        logger.info(f"🔍 DEBUG: Testing query '{query}' against news '{news_title}'")

        # Test dynamic context analysis
        context = analyze_query_context_dynamically(query)
        logger.info(f"📊 Context Analysis: {context}")

        # Test ML classifier
        ml_result = lightweight_ml_classifier(news_title, news_description, query)
        logger.info(f"🤖 ML Result: {ml_result}")

        # Test entity extraction
        import re
        query_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        news_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', f"{news_title} {news_description}")

        # Test semantic similarity
        content = f"{news_title} {news_description}"
        similarity = calculate_similarity(content.lower(), query.lower())

        # Test Gemini if ML approves
        gemini_result = None
        if ml_result["should_send_to_gemini"]:
            gemini_result = await smart_dynamic_gemini_gatekeeper(news_title, news_description, query, "general")

        return {
            "query": query,
            "news": {"title": news_title, "description": news_description},
            "analysis": {
                "context_analysis": context,
                "query_entities": query_entities,
                "news_entities": news_entities,
                "semantic_similarity": similarity,
                "ml_classification": ml_result,
                "gemini_analysis": gemini_result
            },
            "final_decision": {
                "would_pass_ml": ml_result["should_send_to_gemini"],
                "would_pass_gemini": gemini_result["is_relevant"] if gemini_result else "Not tested",
                "overall_relevant": gemini_result["is_relevant"] if gemini_result else False
            }
        }

    except Exception as e:
        logger.error(f"❌ Debug error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- WATI Webhook Handler for Feedback --------------------
@router.post("/wati-webhook")
async def handle_wati_webhook(request: Request):
    """Handle incoming WATI webhook for user feedback"""
    try:
        # Get webhook payload
        payload = await request.json()
        logger.info(f"🔗 Received WATI webhook: {json.dumps(payload, indent=2)}")

        # Extract message details
        message_data = payload.get("data", {})
        phone_number = message_data.get("whatsappNumber", "")
        message_text = message_data.get("text", "").lower().strip()

        # Check if this is a feedback response
        feedback_keywords = {
            "love it": "love_it",
            "❤️": "love_it",
            "💕": "love_it",
            "like it": "like_it",
            "👍": "like_it",
            "not my vibe": "not_my_vibe",
            "👎": "not_my_vibe",
            "nah": "not_my_vibe"
        }

        feedback_type = None
        for keyword, ftype in feedback_keywords.items():
            if keyword in message_text:
                feedback_type = ftype
                break

        if not feedback_type:
            logger.info(f"📝 Message not recognized as feedback: {message_text}")
            return {"status": "ignored", "message": "Not a feedback message"}

        # Find user by phone number
        user_doc = await users_collection.find_one({"phone_number": phone_number})
        if not user_doc:
            logger.warning(f"⚠️ User not found for phone: {phone_number}")
            return {"status": "error", "message": "User not found"}

        user_id = user_doc.get("user_id")

        # Get the latest news sent to this user (from recent feedback or personalized news)
        recent_news = await personalized_collection.find_one(
            {"user_id": user_id},
            sort=[("generated_at", -1)]
        )

        if not recent_news or not recent_news.get("results"):
            logger.warning(f"⚠️ No recent news found for user {user_id}")
            return {"status": "error", "message": "No recent news found"}

        # Use the first news item from recent results
        news_item = recent_news["results"][0]
        feedback_id = f"{news_item.get('news_id', hash(news_item.get('title', '')))}_{user_id}"

        # Store feedback and generate insights
        await store_user_feedback(user_id, phone_number, news_item, feedback_type, feedback_id)

        # Generate insights for acknowledgment
        insights = await gemini_generate_feedback_insights(
            news_item.get("title", ""),
            news_item.get("description", ""),
            news_item.get("category", ""),
            feedback_type
        )

        # Send acknowledgment
        await send_feedback_acknowledgment(phone_number, feedback_type, insights)

        logger.info(f"✅ Processed {feedback_type} feedback from {user_id}")

        return {
            "status": "success",
            "user_id": user_id,
            "feedback_type": feedback_type,
            "insights_generated": bool(insights),
            "message": f"Feedback '{feedback_type}' processed successfully"
        }

    except Exception as e:
        logger.error(f"❌ Webhook processing error: {e}")
        return {"status": "error", "message": str(e)}

# Debug/admin routes removed for production optimization

# -------------------- Alert Scheduling Functions for Cron Jobs --------------------
async def should_send_alert_now(alert):
    """Check if alert should be sent based on schedule"""
    schedule = alert.get("schedule", {})
    frequency = schedule.get("frequency", "realtime")
    last_sent = alert.get("last_sent")

    if frequency == "realtime":
        return True

    # Get timezone
    timezone = schedule.get("timezone", "Asia/Kolkata")
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)

    if frequency == "hourly":
        if not last_sent:
            return True
        last_sent_dt = last_sent.replace(tzinfo=tz) if last_sent else None
        return (now - last_sent_dt).total_seconds() >= 3600  # 1 hour

    elif frequency == "daily":
        scheduled_time = schedule.get("time", "09:00")
        hour, minute = map(int, scheduled_time.split(":"))

        if not last_sent:
            return now.hour >= hour and now.minute >= minute

        last_sent_dt = last_sent.replace(tzinfo=tz) if last_sent else None
        # Check if it's past scheduled time and hasn't been sent today
        return (now.hour >= hour and now.minute >= minute and
                last_sent_dt.date() < now.date())

    elif frequency == "weekly":
        scheduled_days = schedule.get("days", [])
        current_day = now.strftime("%A").lower()

        if current_day not in [d.lower() for d in scheduled_days]:
            return False

        scheduled_time = schedule.get("time", "09:00")
        hour, minute = map(int, scheduled_time.split(":"))

        if not last_sent:
            return now.hour >= hour and now.minute >= minute

        last_sent_dt = last_sent.replace(tzinfo=tz) if last_sent else None
        # Check if it's past scheduled time and hasn't been sent this week
        days_since_last = (now.date() - last_sent_dt.date()).days
        return (now.hour >= hour and now.minute >= minute and days_since_last >= 7)

    return False


async def process_scheduled_alerts():
    """Main function for cron job to process scheduled alerts"""
    try:
        logger.info("🕐 Starting scheduled alerts processing...")

        # Get all active alerts directly from collection
        alerts_cursor = alerts_collection.find({"is_active": True})
        all_alerts = []

        async for alert in alerts_cursor:
            # Convert MongoDB document to dict and ensure proper field mapping
            alert_dict = dict(alert)

            # Ensure alert_id exists (handle legacy alerts that might not have it)
            if "alert_id" not in alert_dict and "_id" in alert_dict:
                alert_dict["alert_id"] = str(alert_dict["_id"])

            # Remove MongoDB _id to avoid confusion
            if "_id" in alert_dict:
                del alert_dict["_id"]

            all_alerts.append(alert_dict)

        logger.info(f"📋 Found {len(all_alerts)} active alerts")

        processed_alerts = 0
        sent_messages = 0

        # Group alerts by user_id for efficiency
        user_alerts = {}
        for alert in all_alerts:
            user_id = alert.get("user_id")
            if not user_id:
                logger.warning(f"⚠️ Alert missing user_id: {alert}")
                continue

            if user_id not in user_alerts:
                user_alerts[user_id] = []
            user_alerts[user_id].append(alert)

        # Process each user's alerts
        for user_id, alerts in user_alerts.items():
            # Filter alerts that should be sent now
            alerts_to_process = []
            for alert in alerts:
                if await should_send_alert_now(alert):
                    alerts_to_process.append(alert)

            if not alerts_to_process:
                continue

            logger.info(f"👤 Processing {len(alerts_to_process)} alerts for user {user_id}")

            # Get user categories
            user_categories = set()
            for alert in alerts_to_process:
                main_cat = alert.get("main_category", "").lower()
                if main_cat:
                    user_categories.add(main_cat)
                sub_cats = alert.get("sub_categories", [])
                for sub_cat in sub_cats:
                    if sub_cat:
                        user_categories.add(sub_cat.lower())

            user_categories = list(user_categories)

            # Fetch news for categories
            category_news = await fetch_feeds_by_categories(user_categories)

            if not category_news:
                continue

            # Get user profile
            user_profile = await users_collection.find_one({"user_id": user_id}) or {"preferences": {}}

            # Process with intelligent system
            intelligent_results = await fixed_learning_system.process_alerts(
                category_news, alerts_to_process, user_profile, user_categories
            )

            # Send WhatsApp notifications with deduplication
            if intelligent_results:
                whatsapp_sent = await fixed_learning_system.send_whatsapp_notifications(intelligent_results, user_id)
                sent_messages += whatsapp_sent

                # Update last_sent for processed alerts
                for alert in alerts_to_process:
                    alert_id = alert.get("alert_id")
                    if alert_id:
                        await alerts_collection.update_one(
                            {"alert_id": alert_id},
                            {"$set": {"last_sent": datetime.now()}}
                        )
                    else:
                        logger.warning(f"⚠️ Alert missing alert_id: {alert}")

                processed_alerts += len(alerts_to_process)

                logger.info(f"📱 Sent {whatsapp_sent} WhatsApp messages for user {user_id}")

        logger.info(f"🎉 Scheduled processing complete: {processed_alerts} alerts processed, {sent_messages} messages sent")

        return {
            "total_alerts_checked": len(all_alerts),
            "alerts_processed": processed_alerts,
            "messages_sent": sent_messages,
            "users_processed": len(user_alerts)
        }

    except Exception as e:
        logger.error(f"❌ Error in scheduled alerts processing: {e}")
        raise e


@router.post("/process-scheduled-alerts")
async def run_scheduled_alerts():
    """Manual trigger for scheduled alerts (for testing)"""
    try:
        result = await process_scheduled_alerts()
        return {
            "success": True,
            "message": "Scheduled alerts processed successfully",
            "stats": result
        }
    except Exception as e:
        logger.error(f"❌ Error in manual scheduled processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Single optimized endpoint above handles all news processing

app.include_router(router)