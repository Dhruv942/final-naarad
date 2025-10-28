from urllib.parse import urlparse
from services.rag_news.google_search import search_google_news
from services.rag_news.config import build_trusted_query, get_trusted_sport_sources
"""
RAG Pipeline Implementation with Three Stages:
1. Preference Parsing (LLM)
2. RAG Retrieval (No ML)
3. LLM Article Filtering + Ranking
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import httpx
from pymongo import UpdateOne, IndexModel, ASCENDING, DESCENDING
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

class AlertPreference(BaseModel):
    """Structured representation of user alert preferences"""
    alert_id: str = Field(..., description="Unique identifier for the alert")
    user_id: str = Field(..., description="User who created the alert")
    category: str = Field(..., description="Main category of interest")
    sub_categories: List[str] = Field(default_factory=list, description="Sub-categories of interest")
    followup_questions: List[str] = Field(default_factory=list, description="Follow-up questions")
    custom_question: str = Field(default="", description="Custom user question")
    canonical_entities: List[str] = Field(default_factory=list, description="Canonical entities from LLM")
    event_conditions: List[str] = Field(default_factory=list, description="Event conditions from LLM")
    contextual_query: str = Field(default="", description="Expanded search query keywords")
    forbidden_topics: List[str] = Field(default_factory=list, description="Topics to exclude")
    trusted_sources: List[str] = Field(default_factory=list, description="Trusted domains for site-scoped search")
    trusted_queries: List[str] = Field(default_factory=list, description="Prebuilt Google queries to run first")
    custom_triggers: List[str] = Field(default_factory=list, description="Derived event triggers")
    tags: List[str] = Field(default_factory=list, description="Selected tags from predefined list")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the alert was created")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When the alert was last updated")

class Article(BaseModel):
    """Article model for storing retrieved content"""
    id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    url: str = Field(..., description="Article URL")
    source: str = Field(..., description="Source of the article")
    published_at: datetime = Field(..., description="Publication timestamp")
    category: str = Field(default="general", description="Article category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class RAGPipeline:
    """RAG Pipeline implementing the three-stage process"""
    
    def __init__(self, db, llm_client, news_fetcher):
        """
        Initialize the RAG Pipeline
        
        Args:
            db: MongoDB database instance
            llm_client: Client for LLM operations
            news_fetcher: Service for fetching news from various sources
        """
        self.db = db
        self.llm = llm_client
        self.news_fetcher = news_fetcher
        self.alerts_collection = db.get_collection("alertspars")
        self.articles_collection = db.get_collection("articles")
        self.notification_queue = db.get_collection("notification_queue")
        
        # Ensure indexes
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary database indexes"""
        self.alerts_collection.create_indexes([
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("last_updated", DESCENDING)]),
            IndexModel([("category", ASCENDING)]),
        ])
        
        self.articles_collection.create_indexes([
            IndexModel([("url", ASCENDING)], unique=True),
            IndexModel([("published_at", DESCENDING)]),
            IndexModel([("source", ASCENDING)]),
        ])
        
        self.notification_queue.create_indexes([
            IndexModel([("alert_id", ASCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("status", ASCENDING)]),
        ])
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an alert through all three stages of the RAG pipeline
        
        Args:
            alert_data: Raw alert data from the user
            
        Returns:
            Dict containing the processing results
        """
        # Stage 1: Parse and store alert preferences
        parsed_alert = await self._parse_alert_preferences(alert_data)
        
        # Stage 2: Fetch relevant articles
        articles = await self._fetch_articles(parsed_alert)
        
        # Stage 3: Filter and rank articles
        results = await self._filter_and_rank_articles(parsed_alert, articles)
        
        # Store and return results
        return await self._store_results(parsed_alert, results)
    
    async def _parse_alert_preferences(self, alert_data: Dict[str, Any]) -> AlertPreference:
        """
        Stage 1: Parse user preferences using LLM
        
        Args:
            alert_data: Raw alert data
            
        Returns:
            Parsed AlertPreference object
        """
        # Check if we already have a parsed version of this alert
        existing = await self.alerts_collection.find_one({"alert_id": alert_data.get("alert_id")})
        if existing:
            return AlertPreference(**existing)
        
        # Prepare prompt for LLM
        prompt = self._build_preference_parsing_prompt(alert_data)
        
        try:
            # Call LLM to parse preferences
            response = await self.llm.generate(prompt)
            parsed_data = self._parse_llm_response(response)
            
            # Create alert preference object; preserve incoming trusted/context fields
            alert_pref = AlertPreference(
                alert_id=alert_data.get("alert_id"),
                user_id=alert_data.get("user_id"),
                category=(alert_data.get("category", "") or "").lower(),
                sub_categories=[s.lower() for s in alert_data.get("sub_categories", [])],
                followup_questions=alert_data.get("followup_questions", []),
                custom_question=alert_data.get("custom_question", ""),
                contextual_query=(parsed_data.get("contextual_query") or alert_data.get("contextual_query") or ""),
                canonical_entities=parsed_data.get("canonical_entities", []),
                event_conditions=parsed_data.get("event_conditions", []),
                forbidden_topics=parsed_data.get("forbidden_topics", []),
                trusted_sources=alert_data.get("trusted_sources", []),
                trusted_queries=alert_data.get("trusted_queries", []),
                custom_triggers=alert_data.get("custom_triggers", []),
                tags=parsed_data.get("tags", [])
            )
            
            # Store in database using Pydantic's .dict() method
            await self.alerts_collection.update_one(
                {"alert_id": alert_pref.alert_id},
                {"$set": alert_pref.dict()},
                upsert=True
            )
            
            return alert_pref
            
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse LLM response as JSON: {je}")
            # Return a basic alert preference with the raw data if parsing fails
            return AlertPreference(
                alert_id=alert_data.get("alert_id"),
                user_id=alert_data.get("user_id"),
                category=(alert_data.get("category", "") or "").lower(),
                sub_categories=[s.lower() for s in alert_data.get("sub_categories", [])],
                followup_questions=alert_data.get("followup_questions", []),
                custom_question=alert_data.get("custom_question", ""),
                contextual_query=alert_data.get("contextual_query", ""),
                trusted_sources=alert_data.get("trusted_sources", []),
                trusted_queries=alert_data.get("trusted_queries", []),
                custom_triggers=alert_data.get("custom_triggers", []),
            )
        except Exception as e:
            logger.error(f"Error parsing alert preferences: {str(e)}")
            # Return a basic alert preference with the raw data if parsing fails
            return AlertPreference(
                alert_id=alert_data.get("alert_id"),
                user_id=alert_data.get("user_id"),
                category=alert_data.get("category", "").lower(),
                sub_categories=[s.lower() for s in alert_data.get("sub_categories", [])],
                followup_questions=alert_data.get("followup_questions", []),
                custom_question=alert_data.get("custom_question", "")
            )
    
    def _build_preference_parsing_prompt(self, alert_data: Dict[str, Any]) -> str:
        """Build the prompt for the LLM to parse alert preferences"""
        return f"""
        Parse the following user alert into structured data:
        
        Category: {alert_data.get('category', '')}
        Sub-categories: {', '.join(alert_data.get('sub_categories', []))}
        Follow-up questions: {', '.join(alert_data.get('followup_questions', []))}
        Custom question: {alert_data.get('custom_question', '')}
        
        Extract the following:
        1. Canonical entities (main people, places, organizations, etc.)
        2. Event conditions (specific events, timeframes, or conditions)
        3. Contextual query (expanded search terms)
        4. Forbidden topics (topics to exclude)
        
        Return as a JSON object with these fields:
        {{
            "canonical_entities": ["entity1", "entity2", ...],
            "event_conditions": ["condition1", "condition2", ...],
            "contextual_query": "expanded search query with synonyms",
            "forbidden_topics": ["topic1", "topic2", ...]
        }}
        """
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback to a more lenient parsing if needed
            logger.warning("Failed to parse LLM response as JSON, attempting recovery")
            # Simple regex to extract JSON-like content
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError("Could not parse LLM response as JSON")
    
    async def _fetch_articles(self, alert: AlertPreference) -> List[Article]:
        """
        Stage 2: Fetch relevant articles from various sources
        
        Args:
            alert: Parsed alert preferences
            
        Returns:
            List of relevant Article objects
        """
        try:
            # Check if we have recent results in cache
            cached = await self._get_cached_articles(alert.alert_id)
            if cached:
                return cached
            
            selected_query = (
                (alert.contextual_query or "").strip()
                or (alert.sub_categories[0] if alert.sub_categories else "")
                or (alert.tags[0] if getattr(alert, "tags", None) else "")
                or (alert.category or "")
            )

            # Debug: show which query and category will be used
            logger.info(
                f"Stage2 Retrieval -> alert_id={alert.alert_id} user_id={alert.user_id} "
                f"category='{alert.category}' selected_query='{selected_query}'"
            )

            articles = []
            seen_urls = set()

            # Helper to coerce any datetime-like value to tz-aware UTC datetime
            def _to_aware_utc(dt_val):
                try:
                    if dt_val is None:
                        return datetime.now(timezone.utc)
                    if isinstance(dt_val, datetime):
                        return dt_val if dt_val.tzinfo is not None else dt_val.replace(tzinfo=timezone.utc)
                    if isinstance(dt_val, str):
                        try:
                            # Try ISO first
                            d = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
                            return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)
                        except Exception:
                            # Fallback: treat as now
                            return datetime.now(timezone.utc)
                except Exception:
                    return datetime.now(timezone.utc)

            # If no precomputed trusted queries, derive simple ones from subcats/entities/followups
            derived_trusted_queries: List[str] = []
            derived_trusted_sources: List[str] = []
            if not alert.trusted_queries:
                # detect sport
                sport = None
                mapping = get_trusted_sport_sources()
                for cand in [(alert.category or "").lower(), *[s.lower() for s in (alert.sub_categories or [])]]:
                    if cand in mapping:
                        sport = cand
                        break
                if sport:
                    spec = mapping.get(sport) or {}
                    tsrc = spec.get("trusted_source")
                    if tsrc:
                        derived_trusted_sources.append(tsrc)
                    # candidates for entities: canonical_entities else followup_questions
                    ent_list = alert.canonical_entities or []
                    if not ent_list:
                        # Use alphanumeric words in followups as rough entities
                        ent_list = [fq for fq in (alert.followup_questions or []) if any(c.isalpha() for c in fq)]
                    for ent in ent_list:
                        q = build_trusted_query(sport, str(ent).strip())
                        if q:
                            derived_trusted_queries.append(q)

            effective_trusted = alert.trusted_queries or derived_trusted_queries
            effective_sources = alert.trusted_sources or derived_trusted_sources

            if effective_trusted:
                for tq in effective_trusted[:5]:
                    try:
                        g_items = await search_google_news(tq, num_results=5, days_back=7, language="en", region="in")
                        for item in g_items:
                            u = item.get("url")
                            if not u or u in seen_urls:
                                continue
                            if effective_sources:
                                host = urlparse(u).netloc.lower()
                                if not any(src in host for src in effective_sources):
                                    continue
                            seen_urls.add(u)
                            try:
                                articles.append(Article(
                                    id=u,
                                    title=item.get("title", ""),
                                    content=item.get("snippet", ""),
                                    url=u,
                                    source=item.get("source", ""),
                                    published_at=_to_aware_utc(item.get("published_at")),
                                    category=alert.category,
                                    metadata={}
                                ))
                            except Exception as e:
                                logger.error(f"Error creating article from trusted query: {e}")
                        # Also try Google News RSS via NewsFetcher for the same trusted query
                        try:
                            rss_items = await self.news_fetcher.fetch("google_news", tq, category=alert.category)
                            for ad in rss_items:
                                u2 = ad.get("url")
                                if not u2 or u2 in seen_urls:
                                    continue
                                if effective_sources:
                                    host2 = urlparse(u2).netloc.lower()
                                    if not any(src in host2 for src in effective_sources):
                                        continue
                                seen_urls.add(u2)
                                try:
                                    articles.append(Article(
                                        id=u2,
                                        title=ad.get("title", ""),
                                        content=ad.get("content", ""),
                                        url=u2,
                                        source=ad.get("source", "unknown"),
                                        published_at=(ad.get("published_at") or ad.get("published_date") or datetime.now(timezone.utc)),
                                        category=ad.get("category", alert.category),
                                        metadata=ad.get("metadata", {})
                                    ))
                                except Exception as e:
                                    logger.error(f"Error creating article from trusted RSS: {e}")
                        except Exception as e:
                            logger.warning(f"Trusted RSS search failed: {e}")
                    except Exception as e:
                        logger.warning(f"Trusted query search failed: {e}")

            sources = ["google_news", "official_rss", "serp_search"]
            # Add Cricbuzz live feed if this looks like a cricket alert
            is_cricket = (alert.category or "").lower() == "cricket" or ("cricket" in [s.lower() for s in (alert.sub_categories or [])])
            if is_cricket:
                sources.insert(0, "cricbuzz_live")
            tasks = [
                self.news_fetcher.fetch(source, selected_query, category=alert.category)
                for source in sources
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error fetching articles: {result}")
                    continue
                for article_data in result:
                    u = article_data.get("url")
                    if not u or u in seen_urls:
                        continue
                    seen_urls.add(u)
                    try:
                        articles.append(Article(
                            id=u,
                            title=article_data.get("title", ""),
                            content=article_data.get("content", ""),
                            url=u,
                            source=article_data.get("source", "unknown"),
                            published_at=_to_aware_utc(article_data.get("published_at") or article_data.get("published_date")),
                            category=article_data.get("category", alert.category),
                            metadata=article_data.get("metadata", {})
                        ))
                    except Exception as e:
                        logger.error(f"Error creating article: {e}")
            
            # Time filter: keep only last 24 hours
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_articles = [a for a in articles if a.published_at >= cutoff]
            logger.info(
                f"Stage2 Retrieval -> alert_id={alert.alert_id} total={len(articles)} last24h={len(recent_articles)}"
            )

            # If nothing found yet, try a non-site Google CSE fallback with entities/category
            if not recent_articles:
                try:
                    base_q = (" ".join(alert.canonical_entities) if alert.canonical_entities else selected_query) or (alert.category or "")
                    fallback_items = await search_google_news(base_q, num_results=10, days_back=7, language="en", region="in")
                    for item in fallback_items:
                        u = item.get("url")
                        if not u:
                            continue
                        try:
                            articles.append(Article(
                                id=u,
                                title=item.get("title", ""),
                                content=item.get("snippet", ""),
                                url=u,
                                source=item.get("source", ""),
                                published_at=_to_aware_utc(item.get("published_at")),
                                category=alert.category,
                                metadata={}
                            ))
                        except Exception:
                            pass
                    # Recompute recency after fallback
                    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                    recent_articles = [a for a in articles if a.published_at >= cutoff]
                    logger.info(
                        f"Stage2 Retrieval Fallback CSE -> total={len(articles)} last24h={len(recent_articles)}"
                    )
                except Exception as e:
                    logger.warning(f"Non-site CSE fallback failed: {e}")

            # Store in database
            if recent_articles:
                await self._store_articles(recent_articles)
            
            return recent_articles
            
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []
    
    async def _get_cached_articles(self, alert_id: str) -> List[Article]:
        """Get cached articles for an alert if they exist and are recent"""
        cache_ttl = timedelta(minutes=0)  # Disable cache during debugging to force fresh retrieval
        
        result = await self.notification_queue.find_one(
            {"alert_id": alert_id},
            sort=[("created_at", -1)]
        )
        
        if result:
            created_at = result.get("created_at")
            if isinstance(created_at, datetime):
                # Coerce to timezone-aware for safe subtraction
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - created_at) < cache_ttl:
                    return [Article(**a) for a in result.get("articles", [])]
        return []
    
    async def _store_articles(self, articles: List[Article]) -> None:
        """Store articles in the database"""
        if not articles:
            return
            
        operations = []
        for article in articles:
            # Article is a Pydantic model, use model_dump() not dataclasses.asdict()
            article_dict = article.model_dump()
            article_dict["_id"] = article_dict.pop("id")
            operations.append(
                UpdateOne(
                    {"_id": article_dict["_id"]},
                    {"$set": article_dict},
                    upsert=True
                )
            )
            
        if operations:
            await self.articles_collection.bulk_write(operations)
    
    async def _filter_and_rank_articles(
        self, 
        alert: AlertPreference, 
        articles: List[Article]
    ) -> Dict[str, Any]:
        """
        Stage 3: Filter and rank articles using LLM
        
        Args:
            alert: Parsed alert preferences
            articles: List of articles to filter and rank
            
        Returns:
            Dict containing filtered and ranked articles
        """
        # Debug: input articles count
        logger.info(
            f"Stage3 Filter -> alert_id={alert.alert_id} category='{alert.category}' "
            f"input_articles={len(articles)}"
        )

        if not articles:
            return {
                "filtered_ranked_articles": [],
                "included_count": 0,
                "excluded_count": 0
            }
        
        # Pin live Cricbuzz items, but only those matching the user's entities/preferences
        try:
            pinned_live = [a for a in articles if (a.source or "").lower() == "cricbuzz live"]
            if pinned_live:
                # Build needles from canonical_entities and followup_questions
                needles_raw = []
                needles_raw.extend(alert.canonical_entities or [])
                needles_raw.extend(alert.followup_questions or [])
                needles = [n.strip().lower() for n in needles_raw if n and any(c.isalpha() for c in n)]
                def _match_live(a: Article) -> bool:
                    title_l = (a.title or "").lower()
                    teams_l = f"{(a.metadata or {}).get('team1','')} {(a.metadata or {}).get('team2','')}".lower()
                    hay = f"{title_l} {teams_l}"
                    return any(n in hay for n in needles) if needles else False
                matched = [a for a in pinned_live if _match_live(a)]
                if matched:
                    top_live = matched[:5]
                    logger.info(
                        f"Stage3 Filter -> Pinned {len(top_live)} Cricbuzz live items matched to entities"
                    )
                    return {
                        "filtered_ranked_articles": top_live,
                        "included_count": len(top_live),
                        "excluded_count": max(0, len(articles) - len(top_live))
                    }
                else:
                    # User asked for only preference-aligned news; do not return unrelated live items
                    logger.info("Stage3 Filter -> No Cricbuzz live matched entities; skipping unrelated lives")
        except Exception:
            pass
        
        # Prepare prompt for LLM
        # Strict pre-filtering: keep only articles that match user's entities/followups; drop forbidden topics
        try:
            needles_raw = []
            needles_raw.extend(alert.canonical_entities or [])
            needles_raw.extend(alert.followup_questions or [])
            needles_raw.extend(getattr(alert, "tags", []) or [])
            needles = [n.strip().lower() for n in needles_raw if n and any(c.isalpha() for c in n)]
            forb = [f.strip().lower() for f in (alert.forbidden_topics or []) if f]

            def _matches_pref(a: Article) -> bool:
                if not needles:
                    return True  # if user gave no entities/followups, don't block
                hay = " ".join([
                    a.title or "",
                    a.content or "",
                    a.source or "",
                    a.url or "",
                    (a.metadata or {}).get("team1", ""),
                    (a.metadata or {}).get("team2", ""),
                    (a.metadata or {}).get("series", ""),
                ]).lower()
                return any(n in hay for n in needles)

            def _contains_forbidden(a: Article) -> bool:
                if not forb:
                    return False
                hay = f"{a.title or ''} {a.content or ''}".lower()
                return any(f in hay for f in forb)

            pre_filtered = [a for a in articles if _matches_pref(a) and not _contains_forbidden(a)]
            if not pre_filtered and needles:
                logger.info("Stage3 Filter -> strict mode: no articles matched user entities; returning empty")
                return {
                    "filtered_ranked_articles": [],
                    "included_count": 0,
                    "excluded_count": len(articles)
                }
            articles = pre_filtered
        except Exception:
            pass

        # Cricket-specific tightening: prefer only live score items
        try:
            is_cricket = (alert.category or "").lower() == "cricket" or ("cricket" in [s.lower() for s in (alert.sub_categories or [])])
            if is_cricket and articles:
                def _is_live_score(a: Article) -> bool:
                    u = (a.url or "").lower()
                    src = (a.source or "").lower()
                    return (
                        src == "cricbuzz live"
                        or "/live-cricket-scores" in u
                        or "/live-scores" in u
                        or "/live-cricket-scorecard" in u
                    )

                live_only = [a for a in articles if _is_live_score(a)]

                # If user requested powerplay/first 5-6 overs or batting first, enforce tighter filters
                cq_text = (alert.custom_question or "") + " " + " ".join(alert.followup_questions or [])
                t = cq_text.lower()
                want_first_overs = any(x in t for x in ["first 5 over", "first five over", "powerplay", "first 6 over", "5 overs", "6 overs"])
                want_batting_first = any(x in t for x in ["batting first", "opt to bat", "won the toss and opted to bat"]) 

                if live_only and (want_first_overs or want_batting_first):
                    def _matches_overs_bf(a: Article) -> bool:
                        md = a.metadata or {}
                        overs = md.get("overs")
                        bf_hint = md.get("batting_first_hint")
                        status = (md.get("status") or "").lower()
                        ok = True
                        if want_first_overs:
                            try:
                                if overs is None:
                                    ok = False
                                else:
                                    ov = float(str(overs))
                                    ok = ov <= 6.0
                            except Exception:
                                ok = False
                        if ok and want_batting_first:
                            ok = bool(bf_hint) or ("opt to bat" in status)
                        return ok

                    live_only = [a for a in live_only if _matches_overs_bf(a)]

                # If we have any live-only items, use them; otherwise fall back to pre-filtered set
                if live_only:
                    articles = live_only
        except Exception:
            pass

        prompt = self._build_filtering_prompt(alert, articles)
        logger.info(
            f"Stage3 Filter -> built prompt for {len(articles)} articles; "
            f"prompt_chars={len(prompt)}"
        )
        
        try:
            # Call LLM for filtering and ranking
            response = await self.llm.generate(prompt)
            logger.info(
                f"Stage3 Filter -> LLM response received; response_chars={len(response) if isinstance(response, str) else 'n/a'}"
            )
            result = self._parse_filtering_response(response, len(articles))
            logger.info(
                f"Stage3 Filter -> included_indices={result.get('included_indices', [])} "
                f"included_count={len(result.get('included_indices', []))}"
            )
            
            # Map back to full article objects
            filtered_articles = []
            for idx in result.get("included_indices", []):
                if 0 <= idx < len(articles):
                    filtered_articles.append(articles[idx])
            
            return {
                "filtered_ranked_articles": filtered_articles,
                "included_count": len(filtered_articles),
                "excluded_count": len(articles) - len(filtered_articles)
            }
            
        except Exception as e:
            logger.error(f"Error filtering/ranking articles: {e}")
            # Return all articles as fallback
            # Apply a simple keyword/entity gatekeeper to avoid off-topic results in fallback
            kw = set(map(str.lower, (
                (alert.sub_categories or []) +
                (alert.canonical_entities or []) +
                ["football", "soccer", "match", "goal", "win", "won", "victory"]
            )))
            def _is_relevant(a: Article) -> bool:
                text = f"{a.title} {a.content}".lower()
                return any(k in text for k in kw)

            filtered = [a for a in articles if _is_relevant(a)]
            fallback_list = (filtered or articles)[:5]
            sample_titles = [a.title for a in fallback_list]
            logger.info(
                f"Stage3 Filter -> Fallback top {min(5, len(articles))} titles={sample_titles}"
            )
            return {
                "filtered_ranked_articles": fallback_list,  # Limit to top 5
                "included_count": len(fallback_list),
                "excluded_count": max(0, len(articles) - len(fallback_list))
            }
    
    def _build_filtering_prompt(self, alert: AlertPreference, articles: List[Article]) -> str:
        """Build the prompt for the LLM to filter and rank articles"""
        # Trim to reduce token usage
        limited = articles[:12]
        articles_json = [
            {
                "title": article.title,
                "content": article.content[:250],  # Short snippet to keep prompt small
                "source": article.source,
                "published_at": article.published_at.isoformat()
            }
            for article in limited
        ]
        
        return f"""
        You are an AI assistant that filters and ranks news articles based on user preferences.
        
        USER ALERT:
        - Category: {alert.category}
        - Sub-categories: {', '.join(alert.sub_categories) if alert.sub_categories else 'None'}
        - Custom question: {alert.custom_question}
        - Canonical entities: {', '.join(alert.canonical_entities) if alert.canonical_entities else 'None'}
        - Event conditions: {', '.join(alert.event_conditions) if alert.event_conditions else 'None'}
        - Forbidden topics: {', '.join(alert.forbidden_topics) if alert.forbidden_topics else 'None'}
        
        ARTICLES TO FILTER AND RANK (in no particular order):
        {json.dumps(articles_json, indent=2, ensure_ascii=False)}
        
        TASK:
        1. Filter out articles that:
           - Don't match the category/sub-categories
           - Don't contain any canonical entities
           - Don't satisfy the event conditions
           - Contain any forbidden topics
        
        2. Rank the remaining articles by relevance to the user's alert.
        
        3. Return a JSON object with this structure:
        {{
            "included_indices": [0, 2, 3],  // Indices of included articles
            "reasoning": "Brief explanation of the filtering and ranking"
        }}
        """
    
    def _parse_filtering_response(self, response: str, total_articles: int) -> Dict[str, Any]:
        """Parse the LLM response from the filtering/ranking step"""
        try:
            result = json.loads(response.strip())
            # Validate indices are within bounds
            included = [
                idx for idx in result.get("included_indices", [])
                if isinstance(idx, int) and 0 <= idx < total_articles
            ]
            return {"included_indices": included}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse filtering response, using all articles")
            return {"included_indices": list(range(min(5, total_articles)))}
    
    async def _store_results(
        self, 
        alert: AlertPreference, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store the RAG pipeline results in notification_queue for sending"""
        # Only store top 3-5 articles for notifications
        top_articles = results.get("filtered_ranked_articles", [])[:5]
        
        if not top_articles:
            logger.info(f"No articles to queue for alert {alert.alert_id}")
            return {"alert_id": alert.alert_id, "status": "no_articles"}
        
        notification_doc = {
            "alert_id": alert.alert_id,
            "user_id": alert.user_id,
            "status": "pending",  # pending, sent, failed
            "created_at": datetime.now(timezone.utc),
            "scheduled_for": datetime.now(timezone.utc),  # Can be updated for scheduled delivery
            "articles": [
                (a.model_dump() if hasattr(a, "model_dump") else (
                    a if isinstance(a, dict) else {}
                ))
                for a in top_articles
            ],
            "stats": {
                "total_processed": results.get("included_count", 0) + results.get("excluded_count", 0),
                "included": results.get("included_count", 0),
                "excluded": results.get("excluded_count", 0),
                "queued": len(top_articles)
            },
            "alert_snapshot": alert.model_dump()
        }
        
        await self.notification_queue.insert_one(notification_doc)
        logger.info(f"Queued {len(top_articles)} articles for alert {alert.alert_id}")
        return notification_doc
    
    async def get_pending_notifications(self, user_id: str = None, alert_id: str = None) -> List[Dict[str, Any]]:
        """
        Get pending notifications from the queue
        
        Args:
            user_id: Optional filter by user ID
            alert_id: Optional filter by alert ID
            
        Returns:
            List of pending notification documents
        """
        query = {"status": "pending"}
        if user_id:
            query["user_id"] = user_id
        if alert_id:
            query["alert_id"] = alert_id
            
        cursor = self.notification_queue.find(query).sort("created_at", -1)
        return await cursor.to_list(None)
    
    async def mark_notification_sent(self, notification_id: str) -> bool:
        """
        Mark a notification as sent
        
        Args:
            notification_id: The notification document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from bson import ObjectId
            result = await self.notification_queue.update_one(
                {"_id": ObjectId(notification_id)},
                {"$set": {"status": "sent", "sent_at": datetime.now(timezone.utc)}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error marking notification as sent: {e}")
            return False


class NewsFetcher:
    """Service for fetching news from various sources using NewsFetcherService"""
    
    def __init__(self):
        """Initialize with actual news fetcher service"""
        try:
            from services.news_fetcher_service import NewsFetcherService
            self.fetcher = NewsFetcherService()
            logger.info("NewsFetcher initialized with NewsFetcherService")
        except ImportError as e:
            logger.warning(f"Could not import NewsFetcherService: {e}")
            self.fetcher = None
    
    async def fetch(self, source: str, query: str, category: str = "general") -> List[Dict[str, Any]]:
        """
        Fetch news articles from the specified source
        
        Args:
            source: Source identifier (google_news, official_rss, serp_search)
            query: Search query
            
        Returns:
            List of article dictionaries
        """
        if not self.fetcher:
            logger.warning("NewsFetcherService not available, returning empty results")
            return []
        
        if source == "google_news":
            return await self._fetch_google_news(query, category)
        elif source == "official_rss":
            return await self._fetch_official_rss(category)
        elif source == "serp_search":
            return await self._fetch_serp_search(query)
        elif source == "cricbuzz_live":
            try:
                # Always fetch all live matches; filtering at this stage may hide relevant items
                return await self.fetcher._fetch_cricbuzz_live(None)
            except Exception as e:
                logger.warning(f"Cricbuzz live fetch failed: {e}")
                return []
        else:
            logger.warning(f"Unknown news source: {source}")
            return []
    
    async def _fetch_google_news(self, query: str, category: str) -> List[Dict[str, Any]]:
        """Fetch news from Google News RSS"""
        try:
            return await self.fetcher._fetch_google_news(query, category=category or "general")
        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")
            return []
    
    async def _fetch_official_rss(self, category: str) -> List[Dict[str, Any]]:
        """Fetch news from official RSS feeds"""
        try:
            return await self.fetcher._fetch_official_rss(category=category or "general")
        except Exception as e:
            logger.error(f"Error fetching official RSS: {e}")
            return []
    
    async def _fetch_serp_search(self, query: str) -> List[Dict[str, Any]]:
        """Fetch news using SERP fallback"""
        try:
            return await self.fetcher._fetch_serp_fallback(query)
        except Exception as e:
            logger.error(f"Error fetching SERP: {e}")
            return []


class LLMClient:
    """Client for interacting with Google's Gemini language model"""
    
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        # Default to shared config key if not provided
        if api_key is None:
            try:
                from services.rag_news.config import GEMINI_API_KEY as _SHARED_KEY
                api_key = _SHARED_KEY
            except Exception:
                api_key = ""
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        self.client = httpx.AsyncClient()
    
    async def _get_available_models(self) -> List[str]:
        """Get list of available models from the API"""
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                params={"key": self.api_key}
            )
            response.raise_for_status()
            models_data = response.json()
            return [model['name'].split('/')[-1] for model in models_data.get('models', [])]
        except Exception as e:
            logger.error(f"Error fetching available models: {str(e)}")
            return []
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Google's Gemini model
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the API call
            
        Returns:
            Generated text
        """
        try:
            # Get available models
            available_models = await self._get_available_models()
            logger.info(f"Available models: {available_models}")
            
            # Try to find a working model
            model_to_try = self.model
            if model_to_try != "gemini-2.5-flash":
                logger.info("Overriding requested model; enforcing gemini-2.5-flash as per requirement")
                model_to_try = "gemini-2.5-flash"
            if model_to_try not in available_models:
                logger.error("Model gemini-2.5-flash not found in available models")
                raise ValueError(f"Required model 'gemini-2.5-flash' not available. Available: {available_models}")
            
            # Prepare the request data
            request_data = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.8,
                    "topK": 40,
                    "maxOutputTokens": 2048,
                    **kwargs
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            logger.info(f"Sending request to model: {model_to_try}")
            response = await self.client.post(
                f"{self.base_url}/models/{model_to_try}:generateContent",
                params={"key": self.api_key},
                json=request_data,
                timeout=30.0
            )
            
            response.raise_for_status()
            data = response.json()

            # Try standard shapes first, then fall back to recursive extraction
            try:
                if 'candidates' in data and data['candidates']:
                    cand = data['candidates'][0]
                    content = cand.get('content') or {}
                    parts = content.get('parts') or []
                    if parts and isinstance(parts, list) and isinstance(parts[0], dict) and 'text' in parts[0]:
                        return parts[0]['text']
            except Exception:
                pass

            if 'content' in data and isinstance(data['content'], dict):
                parts = data['content'].get('parts')
                if parts and isinstance(parts, list) and isinstance(parts[0], dict) and 'text' in parts[0]:
                    return parts[0]['text']

            # Recursive extraction of first 'text' occurrence
            def _extract_text(obj):
                if isinstance(obj, dict):
                    if 'text' in obj and isinstance(obj['text'], str):
                        return obj['text']
                    for v in obj.values():
                        found = _extract_text(v)
                        if found:
                            return found
                elif isinstance(obj, list):
                    for item in obj:
                        found = _extract_text(item)
                        if found:
                            return found
                return None

            extracted = _extract_text(data)
            if extracted:
                return extracted

            logger.error(f"Unexpected response format: {data}")
            raise ValueError("Unexpected response format from Gemini API")
                
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response content: {e.response.text}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Example usage
async def example_usage():
    """Example of how to use the RAG pipeline"""
    from pymongo import MongoClient
    
    # Initialize dependencies
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["rag_pipeline"]
    
    llm_client = LLMClient(api_key="your-api-key")
    news_fetcher = NewsFetcher()
    
    # Create pipeline
    pipeline = RAGPipeline(db, llm_client, news_fetcher)
    
    # Example alert data
    alert_data = {
        "alert_id": "alert_123",
        "user_id": "user_456",
        "category": "sports",
        "sub_categories": ["cricket", "world cup"],
        "followup_questions": ["What are the latest match results?", "Any injuries in the team?"],
        "custom_question": "Show me the latest updates on the Indian cricket team in the World Cup"
    }
    
    try:
        # Process the alert through the pipeline
        results = await pipeline.process_alert(alert_data)
        print(f"Found {len(results['articles'])} relevant articles")
        
        # Get the top 3 articles
        for i, article in enumerate(results["articles"][:3], 1):
            print(f"\n--- Article {i}: {article['title']} ---")
            print(f"Source: {article['source']}")
            print(f"URL: {article['url']}")
            print(f"Published: {article['published_at']}")
            print(article['content'][:200] + "...")
            
    finally:
        # Clean up
        await llm_client.close()
        mongo_client.close()
async def _parse_alert_preferences(self, alert_data: Dict[str, Any]) -> AlertPreference:
    """Parse raw alert data into structured AlertPreference object"""
    logger.info(f"Parsing alert preferences: {alert_data}")

    # Create basic AlertPreference with required fields
    alert_pref = AlertPreference(
        alert_id=alert_data.get("alert_id", str(uuid.uuid4())),
        user_id=alert_data.get("user_id", ""),
        category=alert_data.get("category", alert_data.get("main_category", "")),
        sub_categories=alert_data.get("sub_categories", []),
        followup_questions=alert_data.get("followup_questions", []),
        custom_question=alert_data.get("custom_question", "")
    )

    # Generate prompt for LLM to extract additional information
    prompt = self._build_preference_parsing_prompt(alert_data)
    logger.debug(f"Generated prompt for LLM: {prompt}")

    try:
        # Get response from LLM
        response = await self.llm.generate(prompt)
        logger.debug(f"LLM response: {response}")

        # Parse the LLM response
        parsed_data = self._parse_llm_response(response)
        logger.info(f"Parsed LLM data: {parsed_data}")

        # Update alert_pref with parsed data
        if isinstance(parsed_data, dict):
            alert_pref.canonical_entities = parsed_data.get("canonical_entities", [])
            alert_pref.event_conditions = parsed_data.get("event_conditions", [])
            alert_pref.contextual_query = parsed_data.get("contextual_query", "")
            alert_pref.forbidden_topics = parsed_data.get("forbidden_topics", [])
            
            # If contextual_query is empty, generate one from the alert data
            if not alert_pref.contextual_query:
                alert_pref.contextual_query = self._generate_default_contextual_query(alert_data)

    except Exception as e:
        logger.error(f"Error parsing alert preferences: {str(e)}")
        # Set default values in case of error
        alert_pref.canonical_entities = []
        alert_pref.event_conditions = []
        alert_pref.contextual_query = self._generate_default_contextual_query(alert_data)
        alert_pref.forbidden_topics = []

    # Log the final alert preference
    logger.info(f"Created AlertPreference: {alert_pref}")
    return alert_pref

def _generate_default_contextual_query(self, alert_data: Dict[str, Any]) -> str:
    """Generate a default contextual query from alert data"""
    query_parts = []
    
    if alert_data.get("category"):
        query_parts.append(alert_data["category"])
    
    if alert_data.get("sub_categories"):
        query_parts.extend(alert_data["sub_categories"])
    
    if alert_data.get("followup_questions"):
        query_parts.extend(alert_data["followup_questions"])
    
    if alert_data.get("custom_question"):
        query_parts.append(alert_data["custom_question"])
    
    return " ".join(query_parts) or "latest news"

if __name__ == "__main__":
    asyncio.run(example_usage())
