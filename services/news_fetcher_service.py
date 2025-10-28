"""
Stage 2: News Fetcher Service
Fetches news from multiple sources: Google News RSS, Official RSS, SERP Search
NO ML involved - pure retrieval based on contextual_query
"""

import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import os
import feedparser
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NewsFetcherService:
    """Fetches news from multiple sources for RAG pipeline."""
    
    def __init__(self):
        """Initialize the news fetcher."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = timedelta(minutes=10)
        # RapidAPI (Cricbuzz) config - env override supported
        self.cricbuzz_host = os.getenv("CRICBUZZ_RAPIDAPI_HOST", "cricbuzz-cricket.p.rapidapi.com")
        self.cricbuzz_key = os.getenv("CRICBUZZ_RAPIDAPI_KEY", "c841b69021msh100a3d4ec69b0bdp15d10ajsn2103a84fb644")
        self.cricbuzz_base = os.getenv("CRICBUZZ_RAPIDAPI_BASE", "https://cricbuzz-cricket.p.rapidapi.com")
        
        # Google News Topic RSS feeds (official sections)
        self.topic_feeds = {
            "world": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
            "business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
            "technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
            "entertainment": "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en",
            "sports": "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-IN&gl=IN&ceid=IN:en",
            "science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-IN&gl=IN&ceid=IN:en",
            "health": "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-IN&gl=IN&ceid=IN:en",
            "general": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
        }

        # Additional RSS sources per category (augments Google News topics)
        self.rss_feeds = {
            "sports": [
                # Google News topic will be preferred via topic_feeds
                "https://feeds.feedburner.com/ndtvsports-latest",
            ],
            "cricket": [
                # Specific cricket sources
                "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
            ],
            "technology": [],
            "world": [],
            "business": [],
            "entertainment": [],
            "science": [],
            "health": [],
            "general": []
        }
        
        logger.info("News Fetcher Service initialized")

    def _is_english(self, text: str) -> bool:
        """Heuristic to keep only English-like text.
        Allows common ASCII letters, digits, punctuation and spaces. Accept if >=85% are allowed.
        """
        if not text:
            return True
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()%/-_&+|@#$\n\t\r")
        total = len(text)
        if total == 0:
            return True
        ok = sum(1 for ch in text if ch in allowed)
        return (ok / total) >= 0.85

    def _clean_description(self, raw_html: str) -> str:
        """Strip HTML, collapse whitespace, and trim to ~200 chars."""
        try:
            if not raw_html:
                return ""
            soup = BeautifulSoup(raw_html, "html.parser")
            text = soup.get_text(" ", strip=True)
            text = " ".join(text.split())
            if len(text) > 200:
                text = text[:197].rsplit(" ", 1)[0] + "..."
            return text
        except Exception:
            return (raw_html or "")[:200]
    
    async def fetch_news_for_alert(
        self, 
        alertsparse: Dict[str, Any], 
        max_articles: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles based on alertsparse data.
        
        Args:
            alertsparse: Parsed alert with contextual_query
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        try:
            contextual_query = alertsparse.get("contextual_query", "")
            category = (alertsparse.get("category", "general") or "").lower()
            # Signals for better routing/query building
            canonical_entities = alertsparse.get("canonical_entities", []) or []
            tags = alertsparse.get("tags", []) or []
            followups = alertsparse.get("followup_questions", []) or []
            custom_q = alertsparse.get("custom_question", "") or ""

            logger.info(f"Fetching news for query: {contextual_query[:100]}...")
            
            # Detect live/over-specific intents (English only per requirement)
            def _is_live_intent(txts: List[str]) -> bool:
                hay = " ".join([t for t in txts if isinstance(t, str)]).lower()
                needles = [
                    "live", "batting first", "chasing", "powerplay", "pp", "over", "overs",
                    "last 2 overs", "last two overs", "after 5 overs", "5 overs", "scorecard", "wickets"
                ]
                return any(n in hay for n in needles)

            live_intent = _is_live_intent([contextual_query, custom_q, *followups])

            # Build concise query from entities/tags for better accuracy
            def _build_concise_query() -> str:
                terms: List[str] = []
                for t in canonical_entities:
                    t = (str(t) or "").strip()
                    if not t:
                        continue
                    if " " in t:
                        t = f'"{t}"'
                    terms.append(t)
                for t in tags:
                    t = (str(t) or "").strip()
                    if not t:
                        continue
                    if " " in t:
                        t = f'"{t}"'
                    if t not in terms:
                        terms.append(t)
                # Keep it short
                if not terms:
                    return (category or "general")
                return " ".join(terms[:4])

            concise_query = _build_concise_query()

            # Choose sources: Cricbuzz only for live intents; otherwise Google News + RSS + SERP
            tasks: List[asyncio.Task] = []
            tasks.append(self._fetch_google_news(concise_query or contextual_query, category))
            tasks.append(self._fetch_official_rss(category))
            tasks.append(self._fetch_serp_fallback(concise_query or contextual_query))
            if category == "cricket" and live_intent:
                # Use entities as filters to keep relevant live matches
                team_filters = [str(x).lower() for x in canonical_entities if isinstance(x, str)]
                tasks.append(self._fetch_cricbuzz_live(team_filters))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and deduplicate articles
            all_articles = []
            seen_urls = set()
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Fetch error: {str(result)}")
                    continue
                
                for article in result:
                    url = article.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_articles.append(article)
            
            # Sort by published date (newest first)
            # All dates should be timezone-aware now
            from datetime import timezone
            all_articles.sort(
                key=lambda x: x.get("published_date") or datetime.now(timezone.utc),
                reverse=True
            )
            
            # Limit to max_articles
            final_articles = all_articles[:max_articles]
            
            logger.info(f"Fetched {len(final_articles)} unique articles")
            return final_articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []

    async def _fetch_cricbuzz_live(self, team_filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch live cricket matches from Cricbuzz via RapidAPI.

        team_filters: optional list of strings; if provided, only include matches where
        any filter token appears in team names or series name.
        """
        try:
            url = f"{self.cricbuzz_base}/matches/v1/live"
            headers = {
                "x-rapidapi-host": self.cricbuzz_host,
                "x-rapidapi-key": self.cricbuzz_key,
            }
            logger.info(f"Cricbuzz Live -> GET {url} host={self.cricbuzz_host}")
            resp = await self.http_client.get(url, headers=headers)
            resp.raise_for_status()
            logger.info(f"Cricbuzz Live -> status={resp.status_code}")
            data = resp.json()
            matches = []

            # The response shape groups matches by type; be defensive
            groups = data.get("typeMatches") or []
            now = datetime.now(timezone.utc)
            tf = [t.lower() for t in (team_filters or [])]

            for grp in groups:
                series_list = grp.get("seriesMatches") or []
                for series in series_list:
                    s = series.get("seriesAdWrapper") or {}
                    series_name = (s.get("seriesName") or "").strip()
                    matches_list = s.get("matches") or []
                    for m in matches_list:
                        match_info = m.get("matchInfo") or {}
                        team1_obj = (match_info.get("team1") or {})
                        team2_obj = (match_info.get("team2") or {})
                        team1 = (team1_obj.get("teamName") or "").strip()
                        team2 = (team2_obj.get("teamName") or "").strip()
                        status = (match_info.get("status") or "").strip()
                        match_id = match_info.get("matchId") or match_info.get("matchId")
                        curr_bat_id = match_info.get("currBatTeamId")
                        title = f"{team1} vs {team2} — {series_name}"

                        # Derive live score/overs snippet with batting-first hint
                        score = m.get("matchScore") or {}
                        t1_inn = ((score.get("team1Score") or {}).get("inngs1") or {})
                        t2_inn = ((score.get("team2Score") or {}).get("inngs1") or {})
                        # Pick current batting innings by curr_bat_id
                        current_inn = t1_inn if curr_bat_id == (team1_obj.get("teamId") or team1_obj.get("id")) else (
                            t2_inn if curr_bat_id == (team2_obj.get("teamId") or team2_obj.get("id")) else (t1_inn or t2_inn)
                        )
                        runs = current_inn.get("runs")
                        wkts = current_inn.get("wickets")
                        overs = current_inn.get("overs")
                        # Determine batting-first hint from status text
                        batting_first_hint = "opt to bat" in status.lower()
                        snippet_parts = []
                        if batting_first_hint:
                            snippet_parts.append("Batting first")
                        if runs is not None and overs is not None:
                            # Format wickets if missing
                            wkts_disp = wkts if wkts is not None else 0
                            snippet_parts.append(f"{runs}/{wkts_disp} in {overs} overs")
                            # Powerplay hint for first ~5 overs
                            try:
                                ov_float = float(str(overs).replace(".", "."))
                                if ov_float <= 6.0:
                                    snippet_parts.append("Powerplay in progress")
                            except Exception:
                                pass
                        if not snippet_parts and status:
                            snippet_parts.append(status)
                        content = " — ".join(snippet_parts) if snippet_parts else (status or "Live match update")

                        # Filter if filters provided
                        hay = f"{team1} {team2} {series_name}".lower()
                        if tf and not any(t in hay for t in tf):
                            continue

                        matches.append({
                            "title": title,
                            "url": f"https://www.cricbuzz.com/cricket-match/live-scores/{match_id}" if match_id else "https://www.cricbuzz.com/cricket-match/live-scores",
                            "content": content,
                            "source": "Cricbuzz Live",
                            "category": "cricket",
                            "published_date": now,
                            "fetched_at": now,
                            "metadata": {
                                "match_id": match_id,
                                "team1": team1,
                                "team2": team2,
                                "series": series_name,
                                "status": status,
                                "overs": overs,
                                "runs": runs,
                                "wickets": wkts,
                                "batting_first_hint": batting_first_hint,
                            }
                        })

            logger.info(f"Fetched {len(matches)} live matches from Cricbuzz")
            if matches:
                sample = matches[:2]
                for i, it in enumerate(sample, 1):
                    logger.info(f"Cricbuzz Live sample {i}: title='{it.get('title','')}' url='{it.get('url','')}' status='{(it.get('metadata') or {}).get('status','')}'")
            return matches
        except Exception as e:
            logger.warning(f"Error fetching Cricbuzz live: {e}")
            return []
    
    async def _fetch_google_news(
        self,
        query: str,
        category: str = "general"
    ) -> List[Dict[str, Any]]:
        """Fetch news from Google News RSS.

        Preference order:
        1) If a query is provided, use Google News search RSS
        2) Else if a category is provided, use Google News search RSS with category as the query
        3) Else if category matches a Google News topic, use the topic feed URL
        4) Else use general headlines feed
        """
        try:
            url = None
            cat_key = (category or "").strip().lower()

            # Normalize query from inputs
            q = (query or "").strip()
            if not q and cat_key:
                q = cat_key

            # Build one or two URLs based on available signals
            urls = []
            if q:
                urls.append(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en")
            # If both query and category exist and differ, also fetch category search
            if cat_key and (not q or q.lower() != cat_key):
                urls.append(f"https://news.google.com/rss/search?q={cat_key}&hl=en-IN&gl=IN&ceid=IN:en")
            # If nothing built yet, fall back to topic or general
            if not urls:
                if cat_key in self.topic_feeds:
                    urls.append(self.topic_feeds[cat_key])
                else:
                    urls.append(self.topic_feeds["general"])

            articles = []
            seen = set()
            for u in urls:
                logger.info(
                    f"Google News RSS fetch -> category='{category}', query='{q}', url='{u}'"
                )
                response = await self.http_client.get(u)
                response.raise_for_status()
                feed = feedparser.parse(response.text)
                for entry in feed.entries[:30]:  # Limit per URL
                    title = (entry.get("title", "") or "").strip()
                    link = (entry.get("link", "") or "").strip()
                    if not title or not link:
                        continue
                    key = (title.strip().lower(), link)
                    if key in seen:
                        continue
                    seen.add(key)
                    raw_desc = entry.get("summary", entry.get("description", "")) or ""
                    desc = self._clean_description(raw_desc)
                    if not self._is_english(title + " " + desc):
                        continue
                    image = self._extract_image(entry)
                    article = {
                        "title": title,
                        "url": link,
                        "content": desc,
                        "image": image,
                        "source": "Google News",
                        "category": category,
                        "published_date": self._parse_date(entry.get("published", "")),
                        "fetched_at": datetime.now(timezone.utc)
                    }
                    articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from Google News")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Google News: {str(e)}")
            return []
    
    async def _fetch_official_rss(self, category: str) -> List[Dict[str, Any]]:
        """Fetch news from official RSS feeds based on category."""
        try:
            feeds = self.rss_feeds.get(category, self.rss_feeds["general"])
            
            articles = []
            for feed_url in feeds:
                try:
                    logger.debug(f"Fetching RSS: {feed_url}")
                    response = await self.http_client.get(feed_url)
                    response.raise_for_status()
                    
                    feed = feedparser.parse(response.text)
                    
                    for entry in feed.entries[:20]:
                        title = (entry.get("title", "") or "").strip()
                        link = (entry.get("link", "") or "").strip()
                        if not title or not link:
                            continue
                        raw_desc = entry.get("summary", entry.get("description", "")) or ""
                        desc = self._clean_description(raw_desc)
                        if not self._is_english(title + " " + desc):
                            continue
                        image = self._extract_image(entry)
                        article = {
                            "title": title,
                            "url": link,
                            "content": desc,
                            "image": image,
                            "source": feed.feed.get("title", "RSS Feed"),
                            "category": category,
                            "published_date": self._parse_date(entry.get("published", "")),
                            "fetched_at": datetime.now(timezone.utc)
                        }
                        articles.append(article)
                        
                except Exception as e:
                    logger.warning(f"Error fetching RSS {feed_url}: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from official RSS feeds")
            return articles
            
        except Exception as e:
            logger.error(f"Error in RSS fetch: {str(e)}")
            return []
    
    async def _fetch_serp_fallback(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch news using DuckDuckGo search as SERP fallback.
        This is a free alternative to paid SERP APIs.
        """
        try:
            from duckduckgo_search import DDGS
            
            logger.debug(f"Fetching SERP results for: {query}")
            
            # Use DuckDuckGo search
            ddgs = DDGS()
            results = ddgs.news(query, max_results=20)
            
            articles = []
            for result in results:
                article = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("body", ""),
                    "source": result.get("source", "Web Search"),
                    "category": "general",
                    "published_date": self._parse_date(result.get("date", "")),
                    "fetched_at": datetime.now(timezone.utc)
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from SERP")
            return articles
            
        except Exception as e:
            logger.warning(f"SERP fallback unavailable: {str(e)}")
            return []

    def _extract_image(self, entry: Any) -> Optional[str]:
        """Best-effort extraction of image URL from a feedparser entry."""
        try:
            media_thumb = entry.get('media_thumbnail') or entry.get('media:thumbnail')
            if isinstance(media_thumb, list) and media_thumb:
                url = media_thumb[0].get('url') or media_thumb[0].get('href')
                if url:
                    return url
            media_content = entry.get('media_content') or entry.get('media:content')
            if isinstance(media_content, list) and media_content:
                url = media_content[0].get('url') or media_content[0].get('href')
                if url:
                    return url
            # Sometimes inside links
            if 'links' in entry and isinstance(entry['links'], list):
                for l in entry['links']:
                    if l.get('rel') == 'enclosure' and l.get('type', '').startswith('image/'):
                        return l.get('href')
        except Exception:
            pass
        return None

    async def fetch_category_news(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch Google News RSS Search for each category as per NewsSearchAgent spec.

        Input payload format:
        {
          "news_categories": [ {"label": "Sports", "query": "sports"}, ... ]
        }

        Returns a list of {"category": label, "news": [ ...items... ]}
        """
        results: List[Dict[str, Any]] = []
        try:
            categories = payload.get("news_categories", []) or []
            for cat in categories:
                label = (cat.get("label") or "").strip()
                query = (cat.get("query") or "").strip()

                if not label:
                    continue

                # Build exact search RSS URL without modifying the query
                url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en" if query else self.topic_feeds.get(label.lower(), self.topic_feeds["general"]) 

                items: List[Dict[str, Any]] = []
                try:
                    resp = await self.http_client.get(url)
                    resp.raise_for_status()
                    feed = feedparser.parse(resp.text)

                    seen: set = set()
                    for entry in feed.entries[:10]:
                        title = entry.get("title", "").strip()
                        link = entry.get("link", "").strip()
                        if not title or not link:
                            continue

                        key = (title.lower(), link)
                        if key in seen:
                            continue
                        seen.add(key)

                        desc = entry.get("summary", entry.get("description", "")) or ""
                        pub = self._parse_date(entry.get("published", ""))
                        pub_str = pub.isoformat() if hasattr(pub, "isoformat") else (pub or "")
                        image = self._extract_image(entry)

                        # Filter out obvious ads or placeholders
                        if "advertisement" in title.lower():
                            continue

                        items.append({
                            "title": title,
                            "description": desc,
                            "source": feed.feed.get("title", "Google News"),
                            "published_at": pub_str,
                            "link": link,
                            "image": image
                        })
                except Exception:
                    # On failure, return empty array for this category as per constraints
                    items = []

                results.append({
                    "category": label,
                    "news": items
                })
        except Exception:
            # If overall parsing fails, follow spec: empty arrays per category or empty result
            pass

        return results

    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object (always returns timezone-aware datetime)."""
        try:
            from datetime import timezone
            
            if not date_str:
                return datetime.now(timezone.utc)
            
            # Try parsing with feedparser
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt
            
        except Exception:
            from datetime import timezone
            return datetime.now(timezone.utc)
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Example usage
if __name__ == "__main__":
    async def test():
        fetcher = NewsFetcherService()
        
        alertsparse = {
            "contextual_query": "india cricket team win world cup",
            "category": "cricket"
        }
        
        articles = await fetcher.fetch_news_for_alert(alertsparse, max_articles=10)
        
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   URL: {article['url']}")
        
        await fetcher.close()
    
    asyncio.run(test())
