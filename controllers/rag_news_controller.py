from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urljoin
import hashlib
import os
import httpx

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

def _validate_image_url(img_url: Optional[str], article_url: Optional[str] = None) -> str:
    """
    Validate and normalize image URL. Returns empty string if invalid.
    Only allows http:// or https:// URLs.
    """
    if not img_url or not isinstance(img_url, str):
        return ""
    
    img_url = img_url.strip()
    if not img_url:
        return ""
    
    # Reject data URIs, file://, and other non-http schemes
    if img_url.startswith(('data:', 'file:', 'blob:', '//')):
        return ""
    
    # If relative URL and we have article_url, try to make it absolute
    if not img_url.startswith(('http://', 'https://')):
        if article_url:
            try:
                img_url = urljoin(article_url, img_url)
            except Exception:
                return ""
        else:
            return ""
    
    # Final validation: must be http:// or https://
    parsed = urlparse(img_url)
    if parsed.scheme not in ('http', 'https'):
        return ""
    
    # Must have valid domain
    if not parsed.netloc:
        return ""
    
    return img_url


async def _enrich_images(items: List[Dict[str, Any]]) -> None:
    """Populate image field for items that are missing one, using Google CSE, Google Images, OG scraping, and fallbacks."""
    for item in items:
        try:
            if (item.get("image") or "").strip():
                continue
            title = item.get("title") or ""
            url = item.get("url") or ""
            source = item.get("source") or ""
            category = (item.get("category") or "").lower()
            topic = (item.get("topic") or "").lower()
            keywords = [str(k).lower() for k in (item.get("keywords") or []) if isinstance(k, (str, int, float))]

            final_image = ""

            # 0) Unsplash API (royalty-free) FIRST: context-aware image
            if not final_image:
                try:
                    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY") or os.getenv("UNSPLASH_CLIENT_ID") or "hXFHeM77GyNVp_Vyvh8NqQeFKO6aSIFAd4DceMxeu1c"
                    if unsplash_key:
                        async def _search_unsplash_image(q: str) -> str:
                            try:
                                if not q or len(q.strip()) < 3:
                                    return ""
                                headers = {"Accept-Version": "v1", "Authorization": f"Client-ID {unsplash_key}"}
                                params = {"query": q.strip()[:80], "per_page": 1, "orientation": "landscape"}
                                async with httpx.AsyncClient(timeout=8.0) as client:
                                    resp = await client.get("https://api.unsplash.com/search/photos", headers=headers, params=params)
                                    resp.raise_for_status()
                                    data = resp.json()
                                    results = data.get("results", [])
                                    if results:
                                        urls = (results[0] or {}).get("urls", {})
                                        for key in ["regular", "full", "raw"]:
                                            link = urls.get(key)
                                            if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                                                return link
                            except Exception:
                                return ""
                            return ""

                        unsplash_queries: List[str] = []
                        if title:
                            unsplash_queries.append(title)
                        if topic and title:
                            unsplash_queries.append(f"{topic} {title}")
                        if category and title:
                            unsplash_queries.append(f"{category} {title}")
                        if keywords and title:
                            unsplash_queries.append(f"{' '.join(keywords[:3])} {title}")
                        hay2 = " ".join([title.lower(), topic, category, " ".join(keywords)])
                        if any(k in hay2 for k in ["gold", "bullion", "precious metal"]):
                            unsplash_queries.insert(0, "gold bars")
                        if any(k in hay2 for k in ["stock", "market", "exchange", "finance"]):
                            unsplash_queries.append("stock market board")
                        if any(k in hay2 for k in ["cricket", "match", "wicket", "bat"]):
                            unsplash_queries.append("cricket stadium")
                        if any(k in hay2 for k in ["tech", "technology", "ai", "software", "device"]):
                            unsplash_queries.append("circuit board blue")

                        for uq in unsplash_queries:
                            link = await _search_unsplash_image(uq)
                            if link:
                                v = _validate_image_url(link)
                                if v:
                                    final_image = v
                                    break
                except Exception as _us_err:
                    logger.info(f"Unsplash image search failed: {_us_err}")

            # 1) Google CSE via site/title queries (augmented with topic/keywords)
            try:
                from urllib.parse import urlparse as _urlparse
                host = _urlparse(url).netloc if url else ''
                queries: List[str] = []
                if title:
                    if host:
                        queries.append(f"site:{host} \"{title}\"")
                    if source:
                        queries.append(f"{source} \"{title}\"")
                    queries.append(f"\"{title}\"")
                # Add topic/keywords variants
                if topic and title:
                    queries.append(f"{topic} \"{title}\"")
                if category and title:
                    queries.append(f"{category} \"{title}\"")
                if keywords and title:
                    queries.append(f"{' '.join(keywords[:3])} \"{title}\"")
                # Cricket team-based queries if title has vs
                try:
                    ttl_lower = title.lower()
                    if ' vs ' in ttl_lower or ' vs. ' in ttl_lower:
                        base = title.split(' — ')[0]
                        parts = base.replace(' vs. ', ' vs ').split(' vs ')
                        if len(parts) == 2:
                            team_a, team_b = parts[0].strip(), parts[1].strip()
                            cric_host = host or 'www.cricbuzz.com'
                            queries.extend([
                                f"site:{cric_host} {team_a} {team_b} live",
                                f"site:{cric_host} {team_a} vs {team_b} scorecard",
                                f"cricket {team_a} vs {team_b} image",
                            ])
                except Exception:
                    pass

                for q in queries:
                    cse_results = await search_google_news(q, num_results=5, days_back=30)
                    for r in cse_results:
                        img = r.get('image_url')
                        validated = _validate_image_url(img, article_url=url)
                        if validated:
                            final_image = validated
                            break
                    if final_image:
                        break
            except Exception as _img_err:
                logger.info(f"Image fetch via Google CSE failed: {_img_err}")

            # 2) Google Images API and OG scraping via NewsFetcherService (include topic/keywords)
            if not final_image:
                try:
                    from services.news_fetcher_service import NewsFetcherService as _NFS
                    _nfs = _NFS()
                    img_queries: List[str] = []
                    if title:
                        img_queries.append(title)
                    if source and title:
                        img_queries.append(f"{source} {title}")
                    if topic and title:
                        img_queries.append(f"{topic} {title}")
                    if category and title:
                        img_queries.append(f"{category} {title}")
                    if keywords and title:
                        img_queries.append(f"{' '.join(keywords[:3])} {title}")
                    try:
                        ttl_lower = title.lower()
                        if ' vs ' in ttl_lower or ' vs. ' in ttl_lower:
                            base = title.split(' — ')[0]
                            parts = base.replace(' vs. ', ' vs ').split(' vs ')
                            if len(parts) == 2:
                                team_a, team_b = parts[0].strip(), parts[1].strip()
                                img_queries.append(f"{team_a} {team_b} cricket")
                                img_queries.append(f"{team_a} vs {team_b}")
                    except Exception:
                        pass
                    for iq in img_queries:
                        img = await _nfs._search_google_image(iq)
                        if img:
                            validated = _validate_image_url(img, article_url=url)
                            if validated:
                                final_image = validated
                                break
                    if not final_image and url:
                        scraped = await _nfs._scrape_article_image(url)
                        if scraped:
                            validated = _validate_image_url(scraped, article_url=url)
                            final_image = validated or ""
                    await _nfs.close()
                except Exception as _img2_err:
                    logger.info(f"Image fetch via Google Images/OG failed: {_img2_err}")

            # (Pexels removed as requested)

            # 2.6) Unsplash API (royalty-free) as an additional image source
            if not final_image:
                try:
                    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY") or os.getenv("UNSPLASH_CLIENT_ID") or "hXFHeM77GyNVp_Vyvh8NqQeFKO6aSIFAd4DceMxeu1c"
                    if unsplash_key:
                        async def _search_unsplash_image(q: str) -> str:
                            try:
                                if not q or len(q.strip()) < 3:
                                    return ""
                                headers = {"Accept-Version": "v1", "Authorization": f"Client-ID {unsplash_key}"}
                                params = {"query": q.strip()[:80], "per_page": 1, "orientation": "landscape"}
                                async with httpx.AsyncClient(timeout=8.0) as client:
                                    resp = await client.get("https://api.unsplash.com/search/photos", headers=headers, params=params)
                                    resp.raise_for_status()
                                    data = resp.json()
                                    results = data.get("results", [])
                                    if results:
                                        urls = (results[0] or {}).get("urls", {})
                                        # Prefer regular > full > raw
                                        for key in ["regular", "full", "raw"]:
                                            link = urls.get(key)
                                            if link and isinstance(link, str) and link.startswith(("http://", "https://")):
                                                return link
                            except Exception:
                                return ""
                            return ""

                        unsplash_queries: List[str] = []
                        if title:
                            unsplash_queries.append(title)
                        if topic and title:
                            unsplash_queries.append(f"{topic} {title}")
                        if category and title:
                            unsplash_queries.append(f"{category} {title}")
                        if keywords and title:
                            unsplash_queries.append(f"{' '.join(keywords[:3])} {title}")
                        hay2 = " ".join([title.lower(), topic, category, " ".join(keywords)])
                        if any(k in hay2 for k in ["gold", "bullion", "precious metal"]):
                            unsplash_queries.insert(0, "gold bars")
                        if any(k in hay2 for k in ["stock", "market", "exchange", "finance"]):
                            unsplash_queries.append("stock market board")
                        if any(k in hay2 for k in ["cricket", "match", "wicket", "bat"]):
                            unsplash_queries.append("cricket stadium")
                        if any(k in hay2 for k in ["tech", "technology", "ai", "software", "device"]):
                            unsplash_queries.append("circuit board blue")

                        for uq in unsplash_queries:
                            link = await _search_unsplash_image(uq)
                            if link:
                                v = _validate_image_url(link)
                                if v:
                                    final_image = v
                                    break
                except Exception as _us_err:
                    logger.info(f"Unsplash image search failed: {_us_err}")

            # 3) Fallbacks (category/topic/keyword aware)
            if not final_image:
                # Simple detectors
                text_hay = " ".join([title.lower(), topic, category, " ".join(keywords)])
                is_finance = any(k in text_hay for k in [
                    'usd', 'inr', 'dollar', 'rupee', 'exchange', 'currency', 'rate', 'stock', 'market', 'gold', 'silver', 'oil'
                ])
                is_sport = any(k in text_hay for k in ['sport', 'cricket', 'match', 'vs']) or 'cricbuzz' in source.lower()
                is_tech = any(k in text_hay for k in ['tech', 'technology', 'ai', 'software', 'app', 'device'])

                if is_finance:
                    # Finance/gold placeholder
                    if 'gold' in text_hay:
                        final_image = "https://upload.wikimedia.org/wikipedia/commons/6/69/Gold_Bars.jpg"
                    else:
                        final_image = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Stock_Market_Board.jpg"
                elif is_sport:
                    if 'cricbuzz' in source.lower():
                        final_image = "https://www.cricbuzz.com/a/img/v1/300x170/i1/c170657/cricbuzz-logo.png"
                    else:
                        final_image = "https://upload.wikimedia.org/wikipedia/commons/3/36/Cricket_stumps_bails.jpg"
                elif is_tech:
                    final_image = "https://upload.wikimedia.org/wikipedia/commons/5/5f/Circuit_Board_%28blue%29.jpg"
                else:
                    # Neutral news placeholder
                    final_image = "https://upload.wikimedia.org/wikipedia/commons/6/65/Newspaper_Collage.jpg"

            item["image"] = final_image
        except Exception as e:
            logger.info(f"Image enrichment failed for '{item.get('title','')[:40]}': {e}")

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
        # First check if we have any active alerts for this user
        alerts_count = await alerts_collection.count_documents({
            "user_id": user_id,
            "is_active": True
        })
        
        logger.info(f"Found {alerts_count} active alerts for user {user_id}")
        
        if alerts_count == 0:
            logger.info("No active alerts found for user")
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
                
                # Create AlertPreference from alert
                preference = AlertPreference(
                    user_id=user_id,
                    alert_id=alert_id,
                    category=alert.get("main_category", ""),
                    sub_categories=alert.get("sub_categories", []),
                    followup_questions=alert.get("followup_questions", []),
                    custom_question=alert.get("custom_question", "")
                )
                
                logger.info(f"Processing alert {alert_id}")

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
                        
                        logger.info(f"Trusted queries: {preference.trusted_queries}")
                except Exception as merge_err:
                    logger.warning(f"Failed to merge alertspars for alert {alert_id}: {merge_err}")
                
                # Process alert through RAG pipeline
                try:
                    alert_dict = preference.dict()
                    results = await rag_pipeline.process_alert(alert_dict)
                    
                    if results.get('status') == 'no_articles':
                        logger.info(f"No articles found for alert {alert_id}")
                        continue
                    
                    articles_list = results.get('articles', [])
                    logger.info(f"RAG pipeline returned {len(articles_list)} articles")
                    
                    # Log gatekeeper stats if available
                    gatekeeper_stats = results.get('gatekeeper_stats')
                    if gatekeeper_stats:
                        logger.info(f"Gatekeeper stats: {gatekeeper_stats}")
                    
                except Exception as e:
                    logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
                    continue
                
                # Process and deduplicate articles
                processed_articles = []
                seen_urls = set()
                
                for article_dict in articles_list:
                    # Convert dict to Article object or access as dict
                    if isinstance(article_dict, dict):
                        url = article_dict.get('url')
                        # Extract metadata for generated content
                        metadata = article_dict.get('metadata', {})
                        if not isinstance(metadata, dict):
                            metadata = {}
                        # ALWAYS prefer LLM-generated title/description from metadata
                        title = (metadata.get('generated_title') or '').strip() or article_dict.get('title', '')
                        source = article_dict.get('source', '')
                        published_at = article_dict.get('published_at')
                        content = article_dict.get('content', '') or article_dict.get('snippet', '')
                        # ALWAYS use LLM-generated description (gatekeeper creates this)
                        description = (metadata.get('generated_description') or '').strip()
                        # Enforce LLM-only: skip if LLM title/description missing or too short
                        if not title or len(title) < 3 or not description or len(description) < 10:
                            continue
                        # Fallback only if LLM failed
                        if not description:
                            description = content[:200] if content else (title + " - Latest news update")
                        relevance_score = float(metadata.get('relevance_score', 0) or 0)
                        topic = (metadata.get('topic') or 'general').strip()
                    else:
                        # It's an Article object
                        metadata = article_dict.metadata or {}
                        url = article_dict.url
                        # ALWAYS prefer LLM-generated title/description from metadata
                        title = (metadata.get('generated_title') or '').strip() or article_dict.title
                        source = article_dict.source or ''
                        published_at = article_dict.published_at
                        content = article_dict.content or ''
                        # ALWAYS use LLM-generated description (gatekeeper creates this)
                        description = (metadata.get('generated_description') or '').strip()
                        # Enforce LLM-only: skip if LLM title/description missing or too short
                        if not title or len(title) < 3 or not description or len(description) < 10:
                            continue
                        # Fallback only if LLM failed
                        if not description:
                            if source.lower() == "cricbuzz live":
                                description = f"🏏 {article_dict.title} - Live cricket match updates"
                            else:
                                description = article_dict.content[:200] if article_dict.content else (article_dict.title + " - Latest news update")
                        relevance_score = float(metadata.get('relevance_score', 0) or 0)
                        topic = (metadata.get('topic') or 'general').strip()
                        # Extract image from metadata if available (handled later with validation)
                        # We defer actual selection until before building result_item
                    
                    if not url or not title:
                        continue
                        
                    normalized_url = _normalize_url(url)
                    fingerprint = _generate_fingerprint(normalized_url, title.lower())
                    
                    if fingerprint in seen_urls or await _is_duplicate(user_id, normalized_url, title):
                        continue
                        
                    seen_urls.add(fingerprint)
                    
                    # Mark as sent to avoid duplicates
                    await _mark_as_sent(user_id, normalized_url, title, alert_id)
                    
                    # Clean LLM text helper - EXTREMELY aggressive cleaning
                    def _clean_generated_text(text: str) -> str:
                        if not text:
                            return ""
                        try:
                            import re
                            t = str(text).strip()
                            # Step 1: Remove ALL ellipses first
                            t = re.sub(r"\.\.\.+", "", t)
                            # Step 2: Remove relative-time phrases (anywhere)
                            t = re.sub(r"(?i)\b\d+\s+(?:minutes?|hours?|days?|weeks?|months?)\s+ago\b[\.:]?\s*", "", t)
                            # Step 3: Remove UI fragments from anywhere in text
                            t = re.sub(r"(?i)what\s+is\s+the\s+daily\s+range.*?$", "", t)
                            t = re.sub(r"(?i)today'?s\s*\.\.\.?.*?$", "", t)
                            t = re.sub(r"(?i)read\s+more.*?$", "", t)
                            # Step 4: Remove leading/trailing junk
                            t = re.sub(r"^[\.\s\-]+", "", t)
                            t = re.sub(r"[\.\s\-]+$", "", t)
                            # Step 5: Normalize spaces
                            t = re.sub(r"\s+", " ", t).strip()
                            return t
                        except Exception:
                            return str(text).strip() if text else ""

                    # Clean title
                    title = _clean_generated_text(title)
                    
                    # CRITICAL: Validate financial rates and fix invalid ones (like ₹2025)
                    def _validate_rate_in_text(text: str) -> str:
                        """Remove invalid rates like ₹2025 (years) from financial text"""
                        if not text:
                            return text
                        import re
                        # Remove ₹2025, ₹2024, etc. (years, not rates)
                        text = re.sub(r'₹\s*20\d{2}\b', '', text, flags=re.IGNORECASE)
                        text = re.sub(r'₹\s*([12]\d{3})\b', '', text)  # Any 4-digit starting with 1 or 2
                        # Remove "rate: 2025" patterns
                        text = re.sub(r'(?:rate|price|USD/INR)[\s:]+20\d{2}\b', '', text, flags=re.IGNORECASE)
                        text = re.sub(r'(?:rate|price|USD/INR)[\s:]+([12]\d{3})\b', '', text, flags=re.IGNORECASE)
                        # Clean up
                        text = re.sub(r'\s+', ' ', text).strip()
                        text = re.sub(r':\s*$', '', text)
                        return text
                    
                    # Check if this is financial news
                    is_financial = any(kw in (title + " " + (description or "")).lower() 
                                     for kw in ['usd', 'inr', 'dollar', 'rupee', 'exchange rate', 'currency'])
                    
                    # Validate and fix title if financial
                    if is_financial:
                        title = _validate_rate_in_text(title)
                    
                    # CRITICAL: Use LLM-generated description ONLY (no fallback to raw content if it has fragments)
                    desc_final = ""
                    if description and description.strip():
                        cleaned_desc = _clean_generated_text(description.strip())
                        # Validate financial rates
                        if is_financial:
                            cleaned_desc = _validate_rate_in_text(cleaned_desc)
                        # If cleaned description is meaningful, use it
                        if cleaned_desc and len(cleaned_desc.strip()) >= 10:
                            desc_final = cleaned_desc
                    
                    # Only use content fallback if LLM description doesn't exist AND content is clean
                    if not desc_final and content and content.strip():
                        cleaned_content = _clean_generated_text(content.strip())
                        # Reject content if it contains UI fragments
                        if cleaned_content and not any(frag in cleaned_content.lower() for frag in ['hours ago', 'days ago', 'what is', 'today\'s', '...']):
                            desc_final = cleaned_content[:250]
                    
                    # Last resort: generate simple description
                    if not desc_final or len(desc_final.strip()) < 10:
                        if source.lower() == "cricbuzz live":
                            desc_final = f"🏏 {title} - Live cricket match updates"
                        else:
                            desc_final = f"📌 {title}"
                    
                    # Attach cleaned fields
                    result_item = {
                        "title": title,
                        "description": desc_final,
                        "url": url,
                        "source": source,
                        "published_date": published_at.isoformat() if published_at else "",
                        "scraped_published_date": (metadata.get('scraped_published_date').isoformat() if isinstance(metadata.get('scraped_published_date'), datetime) else metadata.get('scraped_published_date', "")),
                        "snippet": _clean_generated_text(desc_final[:200] if desc_final else (content[:200] if content else "")),
                        "alert_id": alert_id,
                        "category": preference.category,
                        "keywords": preference.sub_categories,
                        "tags": getattr(preference, "tags", []),
                        "topic": topic,
                        "relevance_score": float(relevance_score or 0)
                    }

                    # Defer image selection to a separate enrichment flow
                    result_item["image"] = ""
                    processed_articles.append(result_item)
                
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
                                fallback_desc = fc[:200] if fc else (ft + " - Latest updates")
                                fallback.append({
                                    "title": ft,
                                    "description": fallback_desc,  # Add description field
                                    "url": fu,
                                    "source": fs,
                                    "image": "",  # Add image field
                                    "published_date": fp if isinstance(fp, str) else (fp.isoformat() if fp else ""),
                                    "snippet": fallback_desc[:200],
                                    "alert_id": alert_id,
                                    "category": preference.category,
                                    "keywords": preference.sub_categories,
                                    "tags": getattr(preference, "tags", []),
                                    "topic": "general",
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
        
        # Perform image enrichment as a separate flow
        await _enrich_images(all_articles)

        # Sort articles by relevance score in descending order (handle None safely)
        all_articles.sort(key=lambda x: float(x.get("relevance_score") or 0), reverse=True)
        
        # Immediately send top result via WATI (on GET) for quick user delivery
        try:
            wati_response_payload = None
            if all_articles:
                from controllers.send_controller import send_wati_notification
                top = all_articles[0]
                alert_id_for_send = top.get("alert_id") or (alerts[0].get("_id") if alerts else "")
                alert_payload = {
                    "alert_id": str(alert_id_for_send),
                    "alert_category": top.get("category", ""),
                    "alert_keywords": top.get("keywords", []),
                    "total_articles": 1,
                    "articles": [top]
                }
                wati_resp = await send_wati_notification(user_id, alert_payload)
                wati_response_payload = wati_resp
                if isinstance(wati_resp, dict) and wati_resp.get("status") == "success":
                    logger.info(f"[WATI] message sent: code={wati_resp.get('code')} resp={str(wati_resp.get('response'))[:160]}")
        except Exception as e:
            logger.warning(f"[WATI] send failed: {e}")

        return {
            "status": "success",
            "user_id": user_id,
            "alerts_processed": len(alerts),
            "results": all_articles,
            "wati_response": wati_response_payload
        }
        
    except Exception as e:
        logger.error(f"Error in get_user_news for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
