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
import json

logger = logging.getLogger(__name__)


class NewsFetcherService:
    """Fetches news from multiple sources for RAG pipeline."""
    
    def __init__(self):
        """Initialize the news fetcher."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = timedelta(minutes=10)
        # Google Images search config (try env first, then config.py)
        try:
            from services.rag_news.config import GOOGLE_API_KEY, GOOGLE_CX
            self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY") or GOOGLE_API_KEY
            self.google_cx = os.getenv("GOOGLE_CX") or os.getenv("GOOGLE_SEARCH_CX") or GOOGLE_CX
        except ImportError:
            self.google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_SEARCH_API_KEY")
            self.google_cx = os.getenv("GOOGLE_CX") or os.getenv("GOOGLE_SEARCH_CX")
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
    
    async def _scrape_article_date(self, url: str) -> Optional[datetime]:
        """Best-effort extraction of published date from the article page.
        Parses common meta tags, <time> elements, and JSON-LD (NewsArticle/Breadcrumb/Article) blocks.
        Returns a timezone-aware datetime in UTC when possible.
        """
        try:
            resp = await self.http_client.get(url, follow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # 1) OpenGraph / standard meta tags
            meta_props = [
                ("meta", {"property": "article:published_time"}),
                ("meta", {"name": "article:published_time"}),
                ("meta", {"name": "pubdate"}),
                ("meta", {"name": "publishdate"}),
                ("meta", {"name": "date"}),
                ("meta", {"name": "DC.date.issued"}),
                ("meta", {"itemprop": "datePublished"}),
            ]
            for tag, attrs in meta_props:
                el = soup.find(tag, attrs=attrs)
                if el and (el.get("content") or el.get("value")):
                    dt = self._coerce_datetime(el.get("content") or el.get("value"))
                    if dt:
                        return dt

            # 2) <time datetime="...">
            time_el = soup.find("time")
            if time_el and time_el.get("datetime"):
                dt = self._coerce_datetime(time_el.get("datetime"))
                if dt:
                    return dt

            # 3) JSON-LD blocks
            for script in soup.find_all("script", {"type": "application/ld+json"}):
                try:
                    data = json.loads(script.string or "{}")
                except Exception:
                    continue
                candidates = []
                if isinstance(data, dict):
                    candidates.append(data)
                elif isinstance(data, list):
                    candidates.extend([d for d in data if isinstance(d, dict)])
                for obj in candidates:
                    typ = (obj.get("@type") or obj.get("@graph") or "")
                    # Some sites wrap data in @graph
                    if isinstance(obj.get("@graph"), list):
                        for g in obj.get("@graph"):
                            if isinstance(g, dict):
                                dt = self._coerce_datetime(g.get("datePublished") or g.get("dateModified"))
                                if dt:
                                    return dt
                    dt = self._coerce_datetime(obj.get("datePublished") or obj.get("dateCreated") or obj.get("dateModified"))
                    if dt:
                        return dt

        except Exception:
            return None
        return None

    def _coerce_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Convert various date strings to timezone-aware UTC datetime."""
        if not value or not isinstance(value, str):
            return None
        try:
            # Try ISO first
            d = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d.astimezone(timezone.utc)
        except Exception:
            pass
        try:
            # RFC822 / feed-like
            from email.utils import parsedate_to_datetime
            d = parsedate_to_datetime(value)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d.astimezone(timezone.utc)
        except Exception:
            return None

    def _is_converter_or_history_url(self, url: str) -> bool:
        """Heuristic to skip generic converter/calculator/history pages for rates."""
        if not url:
            return False
        u = url.lower()
        bad_tokens = [
            "converter", "convert", "calculator", "history", "historical",
            "/currency/", "x-rates", "xe.com", "wise.com/us/currency-converter",
        ]
        return any(t in u for t in bad_tokens)

    def _extract_rate_from_text(self, text: str) -> Optional[str]:
        """Find a realistic USD/INR rate (70-100) from short text like title/snippet."""
        if not text:
            return None
        import re
        matches = re.findall(r"(?<!\d)(\d{2}\.\d{1,2}|\d{2})(?!\d)", text)
        for m in matches:
            try:
                v = float(m)
                if 70 <= v <= 100:
                    return f"{v:.2f}" if "." in m else f"{v:.2f}"
            except Exception:
                continue
        return None

    async def _search_google_image(self, query: str) -> Optional[str]:
        """Search Google Images for a query and return first valid image URL"""
        if not self.google_api_key or not self.google_cx:
            logger.debug("Google API key or CX not configured for image search")
            return None
        
        try:
            # Build search query from article title/entities (remove special chars)
            import re
            clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
            clean_query = re.sub(r'\s+', ' ', clean_query)  # Normalize spaces
            if len(clean_query) < 3:
                return None
            
            # Google Custom Search API for Images
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': clean_query[:100],  # Limit query length
                'searchType': 'image',
                'num': 1,  # Only need 1 image
                'safe': 'active',
                'imgSize': 'medium',  # Prefer medium size images
                'imgType': 'photo'  # Prefer photos over graphics
            }
            
            response = await self.http_client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            items = data.get('items', [])
            if items:
                image_url = items[0].get('link', '')
                if image_url and image_url.startswith(('http://', 'https://')):
                    logger.debug(f"Found Google Image for '{query[:40]}': {image_url[:60]}")
                    return image_url
        except httpx.TimeoutException:
            logger.debug(f"Timeout searching Google Images for '{query[:40]}'")
        except httpx.HTTPStatusError as e:
            logger.debug(f"HTTP error {e.response.status_code} searching Google Images")
        except Exception as e:
            logger.debug(f"Error searching Google Images: {type(e).__name__}")
        
        return None
    
    async def _scrape_article_image(self, url: str) -> Optional[str]:
        """Best-effort extraction of image URL from article page.
        Tries OpenGraph, meta tags, JSON-LD, and common image tags.
        Returns first valid image found.
        """
        if not url or not url.startswith(('http://', 'https://')):
            return None
        
        try:
            # Increased timeout for slow sites
            resp = await self.http_client.get(url, follow_redirects=True, timeout=15.0)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            logger.debug(f"Scraping image from {url[:60]}")

            # 1) OpenGraph image (most common)
            og_image = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
            if og_image and og_image.get("content"):
                img_url = og_image.get("content").strip()
                if img_url and img_url.startswith(('http://', 'https://')):
                    logger.debug(f"Found OpenGraph image: {img_url[:80]}")
                    return img_url

            # 2) Twitter card image
            tw_image = soup.find("meta", attrs={"name": "twitter:image"}) or soup.find("meta", attrs={"property": "twitter:image"})
            if tw_image and tw_image.get("content"):
                img_url = tw_image.get("content").strip()
                if img_url and img_url.startswith(('http://', 'https://')):
                    return img_url

            # 3) Standard meta image
            meta_img = soup.find("meta", attrs={"name": "image"})
            if meta_img and meta_img.get("content"):
                img_url = meta_img.get("content").strip()
                if img_url and img_url.startswith(('http://', 'https://')):
                    return img_url

            # 4) JSON-LD (structured data)
            for script in soup.find_all("script", {"type": "application/ld+json"}):
                try:
                    import json
                    data = json.loads(script.string or "{}")
                    candidates = []
                    if isinstance(data, dict):
                        candidates.append(data)
                        # Check @graph if exists
                        if isinstance(data.get("@graph"), list):
                            candidates.extend([d for d in data.get("@graph") if isinstance(d, dict)])
                    elif isinstance(data, list):
                        candidates.extend([d for d in data if isinstance(d, dict)])
                    for obj in candidates:
                        # Check for image in NewsArticle/Article/BlogPosting
                        img = obj.get("image")
                        if isinstance(img, str) and img.startswith(('http://', 'https://')):
                            return img
                        elif isinstance(img, dict):
                            img_url = img.get("url") or img.get("@id") or img.get("contentUrl")
                            if isinstance(img_url, str) and img_url.startswith(('http://', 'https://')):
                                return img_url
                        # Also check thumbnailUrl
                        thumb = obj.get("thumbnailUrl")
                        if isinstance(thumb, str) and thumb.startswith(('http://', 'https://')):
                            return thumb
                except Exception:
                    continue

            # 5) Find first large image in article content (common for finance sites)
            try:
                # Look for img tags in main content areas
                content_areas = soup.find_all(['article', 'main', 'div'], class_=lambda x: x and ('content' in str(x).lower() or 'article' in str(x).lower() or 'body' in str(x).lower()))
                for area in content_areas[:3]:  # Check first 3 content areas
                    imgs = area.find_all('img', src=True)
                    for img in imgs:
                        src = img.get('src', '').strip()
                        if not src:
                            continue
                        # Make absolute URL if relative
                        if src.startswith(('http://', 'https://')):
                            # Check if image seems substantial (not icon/logo)
                            if any(keyword not in src.lower() for keyword in ['icon', 'logo', 'avatar', 'favicon', 'sprite']):
                                # Check dimensions if available
                                width = img.get('width') or ''
                                if not width or (isinstance(width, str) and width.replace('px', '').replace('%', '').isdigit() and int(width.replace('px', '').replace('%', '')) > 100):
                                    return src
                        elif src.startswith('/'):
                            # Relative URL - make absolute
                            from urllib.parse import urljoin
                            abs_url = urljoin(url, src)
                            if abs_url.startswith(('http://', 'https://')):
                                return abs_url
            except Exception:
                pass

        except httpx.TimeoutException:
            logger.debug(f"Timeout scraping image from {url[:50]}")
        except httpx.HTTPStatusError as e:
            logger.debug(f"HTTP error {e.response.status_code} scraping image from {url[:50]}")
        except Exception as e:
            logger.debug(f"Error scraping image from {url[:50]}: {type(e).__name__}")
        return None

    async def _scrape_full_content(self, url: str) -> str:
        """Scrape full article content from URL"""
        try:
            response = await self.http_client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to find main content area (common tags)
            content_tags = soup.find_all(['article', 'main', {'class': ['content', 'article', 'post', 'entry']}])
            if content_tags:
                text = ' '.join([tag.get_text(" ", strip=True) for tag in content_tags])
            else:
                # Fallback: get all text
                text = soup.get_text(" ", strip=True)
            
            text = " ".join(text.split())
            
            # Limit to 2000 chars for processing
            if len(text) > 2000:
                text = text[:2000]
            
            return text
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            return ""
    
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

                        # Build rich description with score details
                        desc_parts = []
                        if batting_first_hint:
                            desc_parts.append("Batting first")
                        if runs is not None and overs is not None:
                            wkts_disp = wkts if wkts is not None else 0
                            desc_parts.append(f"{team1} {runs}/{wkts_disp} in {overs} overs")
                        if status:
                            desc_parts.append(status)
                        rich_description = ". ".join(desc_parts) if desc_parts else content
                        
                        matches.append({
                            "title": title,
                            "url": f"https://www.cricbuzz.com/cricket-match/live-scores/{match_id}" if match_id else "https://www.cricbuzz.com/cricket-match/live-scores",
                            "content": rich_description,  # Use enriched description
                            "source": "Cricbuzz Live",
                            "category": "cricket",
                            "published_date": now,
                            "fetched_at": now,
                            "image": "https://www.cricbuzz.com/a/img/v1/96x96/i1/c170657/cricbuzz-logo.png",  # Cricbuzz logo as default
                            "image_url": "https://www.cricbuzz.com/a/img/v1/96x96/i1/c170657/cricbuzz-logo.png",
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
                for entry in feed.entries[:20]:  # Limit per URL for speed
                    title = (entry.get("title", "") or "").strip()
                    link = (entry.get("link", "") or "").strip()
                    if not title or not link:
                        continue
                    # Skip generic converter/history pages for rate queries
                    if any(k in (q or "").lower() for k in ["usd/inr", "usd inr", "dollar", "rupee", "exchange", "rate"]) and self._is_converter_or_history_url(link):
                        logger.debug(f"Skipping converter/history url: {link}")
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
                    # ALWAYS try scraping from article page if RSS image not found
                    if not image and link:
                        try:
                            image = await self._scrape_article_image(link)
                            if image:
                                logger.debug(f"Scraped image for '{title[:40]}': {image[:60]}")
                        except Exception as img_err:
                            logger.debug(f"Image scrape failed for {link[:50]}: {type(img_err).__name__}")
                    
                    # ALWAYS scrape full content for price/rate articles and extract actual rates
                    article_keywords = (title + " " + desc).lower()
                    if any(kw in article_keywords for kw in ["price", "rate", "stock", "exchange", "rupee", "dollar", "share", "trading", "usd", "inr"]):
                        logger.debug(f"Scraping full content for price article: {title[:60]}")
                        scraped_content = await self._scrape_full_content(link)
                        if scraped_content and len(scraped_content) > 100:
                            # Try to extract actual rate numbers from scraped content
                            import re
                            # Look for USD/INR patterns: "84.50", "₹84.50", "INR 84.50", "1 USD = 84.50 INR", etc.
                            # Also catch "current level of 88.32" pattern
                            rate_patterns = [
                                r'(?:current level|trading at|rate is)\s*(?:of|at)?\s*₹?\s*(\d+\.?\d+)',  # "current level of 88.32"
                                r'(?:USD|usd|dollar)\s*[=:]\s*₹?\s*(\d+\.?\d*)',  # "USD = 84.50"
                                r'₹?\s*(\d+\.?\d*)\s*(?:per|/)',  # "₹84.50 per"
                                r'(\d+\.?\d*)\s*INR',  # "84.50 INR"
                                r'INR\s*(\d+\.?\d*)',  # "INR 84.50"
                                r'(\d+\.?\d+)\s*(?:rupees?|rupee)',  # "84.50 rupee"
                            ]
                            found_rate = None
                            for pattern in rate_patterns:
                                matches = re.findall(pattern, scraped_content, re.IGNORECASE)
                                if matches:
                                    # Found rate! Use the first valid match
                                    rate_value = matches[0]
                                    # Validate: USD/INR should be roughly 80-90, not 2025 or 0.01
                                    try:
                                        rate_float = float(rate_value)
                                        if 70 <= rate_float <= 100:  # Reasonable USD/INR range
                                            found_rate = rate_value
                                            break
                                        elif rate_float > 0.01 and rate_float < 0.02:  # Inverse rate like 0.011268
                                            # Convert to direct: 1/0.011268 ≈ 88.75
                                            found_rate = str(round(1 / rate_float, 2))
                                            break
                                    except:
                                        found_rate = rate_value
                                        break
                            
                            if found_rate:
                                # Found valid rate! Prepend to description
                                desc = f"Current USD/INR rate: ₹{found_rate}. {scraped_content[:400]}"
                                logger.info(f"Extracted valid rate ₹{found_rate} from {link[:80]}")
                            else:
                                # No rate found, use scraped content as-is
                                desc = scraped_content[:800]
                            logger.info(f"Scraped {len(scraped_content)} chars from {link[:80]}")
                    else:
                        # If we didn't scrape or didn’t find, try to extract from title/desc quickly
                        quick_rate = self._extract_rate_from_text(title + " " + desc)
                        if quick_rate:
                            desc = f"Current USD/INR rate: ₹{quick_rate}. {desc}"
                
                article = {
                    "title": title,
                    "url": link,
                    "content": desc,
                    "image": image or "",  # Always include image field (empty if not found)
                    "image_url": image or "",  # Also set image_url for compatibility
                    "source": "Google News",
                    "category": category,
                    "published_date": self._parse_date(entry.get("published", "")),
                    "fetched_at": datetime.now(timezone.utc)
                }
                # Best-effort: scrape page-published date
                try:
                    scraped_dt = await self._scrape_article_date(link)
                    if scraped_dt:
                        article["scraped_published_date"] = scraped_dt
                except Exception:
                    pass
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
                        # Skip generic converter/history pages when looking for rates
                        qh = (category or "") + " " + title
                        if any(k in qh.lower() for k in ["usd/inr", "usd inr", "dollar", "rupee", "exchange", "rate"]) and self._is_converter_or_history_url(link):
                            logger.debug(f"Skipping converter/history url (RSS): {link}")
                            continue
                        raw_desc = entry.get("summary", entry.get("description", "")) or ""
                        desc = self._clean_description(raw_desc)
                        if not self._is_english(title + " " + desc):
                            continue
                        image = self._extract_image(entry)
                        # ALWAYS try scraping from article page if RSS image not found
                        if not image and link:
                            try:
                                image = await self._scrape_article_image(link)
                                if image:
                                    logger.debug(f"Scraped image for '{title[:40]}': {image[:60]}")
                            except Exception as img_err:
                                logger.debug(f"Image scrape failed for {link[:50]}: {type(img_err).__name__}")
                        
                        # Scrape and extract rates for price-related articles
                        article_keywords = (title + " " + desc).lower()
                        if any(kw in article_keywords for kw in ["price", "rate", "exchange", "rupee", "dollar", "usd", "inr"]):
                            try:
                                scraped_content = await self._scrape_full_content(link)
                                if scraped_content and len(scraped_content) > 100:
                                    import re
                                    # Look for USD/INR rate patterns
                                    rate_patterns = [
                                        r'(?:USD|usd|dollar)\s*[=:]\s*₹?\s*(\d+\.?\d*)',
                                        r'₹?\s*(\d+\.?\d*)\s*(?:per|/)',
                                        r'(\d+\.?\d*)\s*INR',
                                        r'INR\s*(\d+\.?\d*)',
                                    ]
                                    for pattern in rate_patterns:
                                        matches = re.findall(pattern, scraped_content, re.IGNORECASE)
                                        if matches:
                                            rate_value = matches[0]
                                            desc = f"Current rate: ₹{rate_value} per USD. {scraped_content[:500]}"
                                            logger.info(f"Extracted rate ₹{rate_value} from RSS article {link[:80]}")
                                            break
                                    else:
                                        desc = scraped_content[:800]
                            except Exception:
                                pass
                        
                        article = {
                            "title": title,
                            "url": link,
                            "content": desc,
                            "image": image or "",
                            "image_url": image or "",
                            "source": feed.feed.get("title", "RSS Feed"),
                            "category": category,
                            "published_date": self._parse_date(entry.get("published", "")),
                            "fetched_at": datetime.now(timezone.utc)
                        }
                        try:
                            scraped_dt = await self._scrape_article_date(link)
                            if scraped_dt:
                                article["scraped_published_date"] = scraped_dt
                        except Exception:
                            pass
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
                url = result.get("url", "")
                body = result.get("body", "") or ""
                # Skip generic converter/history urls for rate queries
                if any(k in (query or "").lower() for k in ["usd/inr", "usd inr", "dollar", "rupee", "exchange", "rate"]) and self._is_converter_or_history_url(url):
                    logger.debug(f"Skipping converter/history url (SERP): {url}")
                    continue
                
                # Scrape and extract rates for price-related articles
                title_body = (result.get("title", "") + " " + body).lower()
                if any(kw in title_body for kw in ["price", "rate", "exchange", "rupee", "dollar", "usd", "inr"]):
                    try:
                        scraped_content = await self._scrape_full_content(url)
                        if scraped_content and len(scraped_content) > 100:
                            import re
                            rate_patterns = [
                                r'(?:USD|usd|dollar)\s*[=:]\s*₹?\s*(\d+\.?\d*)',
                                r'₹?\s*(\d+\.?\d*)\s*(?:per|/)',
                                r'(\d+\.?\d*)\s*INR',
                                r'INR\s*(\d+\.?\d*)',
                            ]
                            for pattern in rate_patterns:
                                matches = re.findall(pattern, scraped_content, re.IGNORECASE)
                                if matches:
                                    rate_value = matches[0]
                                    body = f"Current rate: ₹{rate_value} per USD. {scraped_content[:500]}"
                                    logger.info(f"Extracted rate ₹{rate_value} from SERP result {url[:80]}")
                                    break
                            else:
                                body = scraped_content[:800]
                    except Exception:
                        pass
                else:
                    # Quick extraction from title/body if scrape isn't triggered
                    quick_rate = self._extract_rate_from_text(result.get("title", "") + " " + body)
                    if quick_rate:
                        body = f"Current USD/INR rate: ₹{quick_rate}. {body}"
                
                # Try to extract image from article page
                image_url = None
                try:
                    if url:
                        image_url = await self._scrape_article_image(url)
                        if image_url:
                            logger.debug(f"Scraped image for SERP '{result.get('title', '')[:40]}': {image_url[:60]}")
                except Exception as img_err:
                    logger.debug(f"Image scrape failed for SERP {url[:50]}: {type(img_err).__name__}")
                    image_url = None
                
                article = {
                    "title": result.get("title", ""),
                    "url": url,
                    "content": body,
                    "image": image_url or "",
                    "image_url": image_url or "",
                    "source": result.get("source", "Web Search"),
                    "category": "general",
                    "published_date": self._parse_date(result.get("date", "")),
                    "fetched_at": datetime.now(timezone.utc)
                }
                try:
                    scraped_dt = await self._scrape_article_date(url)
                    if scraped_dt:
                        article["scraped_published_date"] = scraped_dt
                except Exception:
                    pass
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from SERP")
            return articles
            
        except Exception as e:
            logger.warning(f"SERP fallback unavailable: {str(e)}")
            return []

    def _extract_image(self, entry: Any) -> Optional[str]:
        """Best-effort extraction of image URL from a feedparser entry."""
        try:
            # Try media:thumbnail first (Google News RSS)
            media_thumb = entry.get('media_thumbnail') or entry.get('media:thumbnail')
            if isinstance(media_thumb, list) and media_thumb:
                url = media_thumb[0].get('url') or media_thumb[0].get('href')
                if url and url.startswith(('http://', 'https://')):
                    logger.debug(f"Found RSS thumbnail: {url[:60]}")
                    return url
            
            # Try media_content (alternative RSS format)
            media_content = entry.get('media_content') or entry.get('media:content')
            if isinstance(media_content, list):
                for media in media_content:
                    if isinstance(media, dict):
                        url = media.get('url') or media.get('href')
                        if url and url.startswith(('http://', 'https://')):
                            # Check if it's an image type
                            media_type = media.get('type', '').lower()
                            if 'image' in media_type or not media_type:
                                logger.debug(f"Found RSS media image: {url[:60]}")
                                return url
            
            # Try links with rel="enclosure" or type="image"
            links = entry.get('links', [])
            for link in links:
                if isinstance(link, dict):
                    rel = link.get('rel', '').lower()
                    link_type = link.get('type', '').lower()
                    href = link.get('href', '')
                    if (rel in ['enclosure', 'image'] or 'image' in link_type) and href and href.startswith(('http://', 'https://')):
                        logger.debug(f"Found RSS link image: {href[:60]}")
                        return href
        except Exception as e:
            logger.debug(f"Error extracting RSS image: {type(e).__name__}")
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
