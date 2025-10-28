from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode
import hashlib
import os

from core.rag_system_enhanced import RAGSystem, Document, UserProfile, SentenceTransformerEmbedding
from core.rag_pipeline import RAGPipeline, AlertPreference, Article, NewsFetcher, LLMClient
from services.rag_news.google_search import search_google_news
from services.rag_news.neural_reranker import NeuralReranker
from services.rag_news.config import GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_CX
from db.mongo import alerts_collection, db

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize RAG components
embedding_model = SentenceTransformerEmbedding()
rag_system = RAGSystem(
    db=db,
    embedding_model=embedding_model,
    gemini_api_key=GEMINI_API_KEY
)

# Initialize LLM Client (use only gemini-2.5-flash)
llm_client = LLMClient(api_key=GEMINI_API_KEY, model="gemini-2.5-flash")

# Initialize News Fetcher
news_fetcher = NewsFetcher()

# Initialize RAG Pipeline
rag_pipeline = RAGPipeline(
    db=db,
    llm_client=llm_client,
    news_fetcher=news_fetcher  # Use NewsFetcher for fetching articles
)

# Initialize neural reranker
neural_reranker = NeuralReranker()

def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    try:
        if not url:
            return ""
        parsed = urlparse(url)
        host = parsed.netloc.lower().lstrip('www.').lstrip('m.')
        path = (parsed.path or "").rstrip("/").lower()
        
        # Keep only important query parameters
        keep_params = {"id", "p", "v", "storyid", "article"}
        query = parse_qsl(parsed.query, keep_blank_values=True)
        filtered_query = [(k, v) for k, v in query if k.lower() in keep_params]
        
        # Rebuild URL
        query_str = f"?{urlencode(filtered_query)}" if filtered_query else ""
        return f"{host}{path}{query_str}"
    except Exception:
        return url or ""

def _generate_fingerprint(url: str, title: str) -> str:
    """Generate a fingerprint for deduplication."""
    domain = (urlparse(url).netloc or "").lower().lstrip('www.').lstrip('m.')
    title_lower = (title or "").strip().lower()
    return hashlib.sha1(f"{domain}|{title_lower}".encode()).hexdigest()

async def _is_duplicate(user_id: str, url: str, title: str, dedup_hours: int = 1) -> bool:
    """Check if a news item has already been sent to the user."""
    try:
        coll = db.get_collection("sent_notifications")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=dedup_hours)
        
        query = {
            "user_id": user_id,
            "created_at": {"$gte": cutoff.isoformat()},
            "$or": [
                {"url": _normalize_url(url)},
                {"raw_url": url},
                {"title_lower": (title or "").strip().lower()},
                {"fp": _generate_fingerprint(url, title)}
            ]
        }
        
        return await coll.count_documents(query) > 0
    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")
        return False

async def _mark_as_sent(user_id: str, url: str, title: str, alert_id: str) -> None:
    """Mark a news item as sent to the user."""
    try:
        coll = db.get_collection("sent_notifications")
        doc = {
            "user_id": user_id,
            "url": _normalize_url(url),
            "raw_url": url,
            "title": title,
            "title_lower": (title or "").strip().lower(),
            "fp": _generate_fingerprint(url, title),
            "alert_id": alert_id,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await coll.update_one(
            {"user_id": user_id, "url": doc["url"]},
            {"$set": doc},
            upsert=True
        )
    except Exception as e:
        logger.warning(f"Failed to mark as sent: {e}")

@router.get("/news/{user_id}")
async def get_user_news(user_id: str) -> Dict[str, Any]:
    """
    Get personalized news for a user based on their alerts using the RAG pipeline.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        Dict containing the news results
    """
    try:
        # Get user's active alerts from alerts collection
        print(f"\n=== DEBUG: Fetching alerts for user {user_id} ===")
        
        # First check if we have any active alerts for this user
        alerts_count = await alerts_collection.count_documents({
            "user_id": user_id,
            "is_active": True
        })
        
        print(f"Found {alerts_count} active alerts for user")
        
        if alerts_count == 0:
            print("No active alerts found for user")
            return {
                "status": "success",
                "user_id": user_id,
                "alerts_processed": 0,
                "results": [],
                "message": "No active alerts found for user"
            }
            
        # Get all active alerts for processing
        alerts = await alerts_collection.find({
            "user_id": user_id,
            "is_active": True
        }).to_list(None)
        
        all_articles = []
        
        for alert in alerts:
            try:
                alert_id = str(alert["_id"])
                
                # Debug print alert data
                print(f"\n=== Processing Alert {alert_id} ===")
                print(f"Alert data: {alert}")
                
                # Create AlertPreference from alert
                preference = AlertPreference(
                    user_id=user_id,
                    alert_id=alert_id,
                    category=alert.get("main_category", ""),  # Use main_category as category
                    sub_categories=alert.get("sub_categories", []),
                    followup_questions=alert.get("followup_questions", []),
                    custom_question=alert.get("custom_question", "")
                )
                
                print(f"Created AlertPreference: {preference}")

                # Merge parsed preferences from alertspars (if available)
                try:
                    parsed = await db.get_collection("alertspars").find_one({"alert_id": alert_id})
                    if parsed:
                        # contextual_query
                        cq = parsed.get("contextual_query")
                        if isinstance(cq, str) and cq.strip():
                            preference.contextual_query = cq.strip()
                        # canonical_entities
                        ce = parsed.get("canonical_entities")
                        if isinstance(ce, list):
                            preference.canonical_entities = [str(x) for x in ce if x is not None]
                        # event_conditions (coerce to strings if objects)
                        ec = parsed.get("event_conditions")
                        if isinstance(ec, list):
                            preference.event_conditions = [
                                (e if isinstance(e, str) else str(e)) for e in ec if e is not None
                            ]
                        # forbidden_topics
                        ft = parsed.get("forbidden_topics")
                        if isinstance(ft, list):
                            preference.forbidden_topics = [str(x) for x in ft if x is not None]
                        # trusted fields
                        ts = parsed.get("trusted_sources")
                        if isinstance(ts, list):
                            preference.trusted_sources = [str(x) for x in ts if x]
                        tq = parsed.get("trusted_queries")
                        if isinstance(tq, list):
                            preference.trusted_queries = [str(x) for x in tq if x]
                        ct = parsed.get("custom_triggers")
                        if isinstance(ct, list):
                            preference.custom_triggers = [str(x) for x in ct if x]
                        tg = parsed.get("tags")
                        if isinstance(tg, list):
                            preference.tags = [str(x) for x in tg if x]
                        print("Merged alertspars into preference (contextual_query/entities/conditions/topics/trusted)")
                except Exception as merge_err:
                    logger.warning(f"Failed to merge alertspars for alert {alert_id}: {merge_err}")
                
                # Process alert through RAG pipeline
                print("Processing alert through RAG pipeline...")
                try:
                    # Convert AlertPreference to dict before processing
                    alert_dict = preference.dict()
                    results = await rag_pipeline.process_alert(alert_dict)
                    
                    # Check if we got results
                    if results.get('status') == 'no_articles':
                        print(f"No articles found for alert {alert_id}")
                        continue
                    
                    articles_list = results.get('articles', [])
                    print(f"RAG pipeline returned {len(articles_list)} articles")
                    
                except Exception as e:
                    print(f"Error in RAG pipeline: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Process and deduplicate articles
                processed_articles = []
                seen_urls = set()
                
                for article_dict in articles_list:
                    # Convert dict to Article object or access as dict
                    if isinstance(article_dict, dict):
                        url = article_dict.get('url')
                        title = article_dict.get('title')
                        source = article_dict.get('source', '')
                        published_at = article_dict.get('published_at')
                        content = article_dict.get('content', '')
                    else:
                        # It's an Article object
                        url = article_dict.url
                        title = article_dict.title
                        source = article_dict.source or ''
                        published_at = article_dict.published_at
                        content = article_dict.content
                    
                    if not url or not title:
                        continue
                        
                    normalized_url = _normalize_url(url)
                    fingerprint = _generate_fingerprint(normalized_url, title.lower())
                    
                    if fingerprint in seen_urls or await _is_duplicate(user_id, normalized_url, title):
                        continue
                        
                    seen_urls.add(fingerprint)
                    
                    # Mark as sent to avoid duplicates
                    await _mark_as_sent(user_id, normalized_url, title, alert_id)
                    
                    processed_articles.append({
                        "title": title,
                        "url": url,
                        "source": source,
                        "published_date": published_at.isoformat() if published_at else "",
                        "snippet": content[:200] if content else "",
                        "alert_id": alert_id,
                        "category": preference.category,
                        "keywords": preference.sub_categories,
                        "tags": getattr(preference, "tags", []),
                        "relevance_score": float((article_dict.get('relevance_score') if isinstance(article_dict, dict) else getattr(article_dict, 'relevance_score', 0)) or 0)
                    })
                
                if not processed_articles and articles_list:
                    fallback = []
                    for ad in (articles_list[:5] if isinstance(articles_list, list) else []):
                        if isinstance(ad, dict):
                            fu = ad.get('url')
                            ft = ad.get('title')
                            fs = ad.get('source', '')
                            fp = ad.get('published_at')
                            fc = ad.get('content', '')
                            if fu and ft:
                                fallback.append({
                                    "title": ft,
                                    "url": fu,
                                    "source": fs,
                                    "published_date": fp if isinstance(fp, str) else (fp.isoformat() if fp else ""),
                                    "snippet": fc[:200] if fc else "",
                                    "alert_id": alert_id,
                                    "category": preference.category,
                                    "keywords": preference.sub_categories,
                                    "tags": getattr(preference, "tags", []),
                                    "relevance_score": 0.0
                                })
                    processed_articles.extend(fallback)

                all_articles.extend(processed_articles)
                
                # Update user profile with interaction (if articles were found)
                if articles_list:
                    query_text = preference.contextual_query or preference.custom_question or " ".join(preference.sub_categories)
                    # Note: record_search_interaction expects Document objects, skipping for now
                    # Can be implemented later when needed
                
            except Exception as e:
                logger.error(f"Error processing alert {alert.get('_id')}: {str(e)}", exc_info=True)
                continue
        
        # Sort articles by relevance score in descending order (handle None safely)
        all_articles.sort(key=lambda x: float(x.get("relevance_score") or 0), reverse=True)
        
        return {
            "status": "success",
            "user_id": user_id,
            "alerts_processed": len(alerts),
            "results": all_articles
        }
        
    except Exception as e:
        logger.error(f"Error in get_user_news for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
