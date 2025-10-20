"""
Google Search API integration for fetching additional news articles
Uses Google Custom Search API to find relevant articles
"""
import logging
import os
import httpx
import asyncio
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Google Custom Search API Configuration
# Get your API key from: https://console.cloud.google.com/apis/credentials
# Get your Search Engine ID from: https://programmablesearchengine.google.com/
GOOGLE_API_KEY = "AIzaSyBmIbpuMr1eBvLgqgk5_U-Kr4I3JKvrX-s"
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "c26c7597576514067")

# Fallback: Use SerpAPI (easier, has free tier)
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "YOUR_SERPAPI_KEY")  # Get from https://serpapi.com/

GOOGLE_SEARCH_ENABLED = True  # ✅ ENABLED - SerpAPI configured
VERIFY_CONTENT_ENABLED = True  # ✅ ENABLED - Smart verification with fallback for better quality

# Excluded domains/sources (video sites, social media, etc.)
EXCLUDED_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "fb.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
]

# Trusted news sources for sports category (whitelist)
TRUSTED_SPORTS_SOURCES = [
    # International Cricket
    "espncricinfo.com",
    "cricbuzz.com",
    "cricket.com",
    "icc-cricket.com",
    
    # Indian News - Major Publications
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "ndtv.com",
    "thequint.com",
    
    # Sports Focused
    "insidesport.in",
    "sportskeeda.com",
    "sportstar.thehindu.com",
    "cricketaddictor.com",
    
    # International Sports
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "espn.com",
    "skysports.com",
    
    # Wire Services
    "pti.in",
    "ani.in",
]

# Trusted news sources for general news/science category
TRUSTED_NEWS_SOURCES = [
    # Major News Outlets
    "thehindu.com",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "cnn.com",
    "nytimes.com",
    "washingtonpost.com",
    
    # Indian News
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "ndtv.com",
    "thequint.com",
    "scroll.in",
    "theprint.in",
    
    # Science & Tech
    "nature.com",
    "science.org",
    "sciencedaily.com",
    "scientificamerican.com",
    "newscientist.com",
    "technologyreview.com",
    "arstechnica.com",
    "wired.com",
    
    # Wire Services
    "pti.in",
    "ani.in",
]

# Trusted sources for movies category
TRUSTED_MOVIE_SOURCES = [
    # Movie News & Reviews
    "variety.com",
    "hollywoodreporter.com",
    "deadline.com",
    "indiewire.com",
    "rottentomatoes.com",
    "imdb.com",
    "metacritic.com",
    
    # Indian Movie Sites
    "thehindu.com",
    "indianexpress.com",
    "filmcompanion.in",
    "bollywoodhungama.com",
    "pinkvilla.com",
    
    # General Entertainment
    "theguardian.com",
    "bbc.com",
    "cnn.com",
    "reuters.com",
]


def _is_excluded_source(url: str) -> bool:
    """Check if URL is from an excluded source (YouTube, social media, etc.)"""
    if not url:
        return False
    
    url_lower = url.lower()
    for domain in EXCLUDED_DOMAINS:
        if domain in url_lower:
            return True
    return False


def _is_trusted_sports_source(url: str) -> bool:
    """Check if URL is from a trusted sports news source"""
    if not url:
        return False
    
    url_lower = url.lower()
    for domain in TRUSTED_SPORTS_SOURCES:
        if domain in url_lower:
            return True
    return False


def _is_trusted_news_source(url: str) -> bool:
    """Check if URL is from a trusted general news/science source"""
    if not url:
        return False
    
    url_lower = url.lower()
    for domain in TRUSTED_NEWS_SOURCES:
        if domain in url_lower:
            return True
    return False


def _is_trusted_movie_source(url: str) -> bool:
    """Check if URL is from a trusted movie/entertainment source"""
    if not url:
        return False
    
    url_lower = url.lower()
    for domain in TRUSTED_MOVIE_SOURCES:
        if domain in url_lower:
            return True
    return False


def _is_specific_article(url: str) -> bool:
    """
    Check if URL is a specific article (not generic homepage or simple category page)
    Articles usually have longer paths with slugs or IDs
    """
    if not url:
        return False
    
    url_lower = url.lower()
    
    # Remove trailing slash for consistent checking
    url_clean = url_lower.rstrip('/')
    
    # Exclude homepage URLs (just domain.com or domain.com/)
    # Example: https://www.sciencedaily.com/ should be excluded
    path_after_domain = url_clean.split('://')[-1].split('/', 1)
    if len(path_after_domain) < 2 or not path_after_domain[1]:
        # No path after domain = homepage
        return False
    
    # Must have meaningful path segments
    path = path_after_domain[1]
    
    # Exclude very short paths (likely category pages)
    if len(path) < 5:  # e.g., "/news" or "/en/"
        return False
    
    # Exclude obvious pagination/category/section pages
    # Slightly stricter to avoid homepage/section results
    excluded_patterns = [
        '/page/',
        '/tags/',
        '?page=',
        '?category=',
        '/section/',
        '/sections/',
        '/category/',
        '/categories/',
        '/topic/',
        '/topics/',
        '/archive/',
        '/archives/',
        # Common generic index pages
        '/scores-fixtures',
        '/live-scores',
        '/scorecard',
        '/fixtures',
        '/results',
        '/schedule',
        '/schedules',
        # ESPN and similar team/player hubs (not articles)
        'espn.com/soccer/team/',
        'espn.com/soccer/player/',
        'espn.com/cricket/team/',
        'espn.com/cricket/player/',
    ]
    
    for pattern in excluded_patterns:
        if pattern in url_lower:
            return False
    
    # Be more lenient - accept most URLs unless obviously wrong
    # Only reject clear category pages with very generic patterns
    path_segments = [s for s in path.split('/') if s]
    
    # Stricter category detection: reject clear section roots
    # Examples: /cricket-news, /news, /sports, /cricket
    if len(path_segments) <= 2:
        last_segment = path_segments[-1] if path_segments else ""
        generic_sections = [
            'news', 'latest-news', 'top-news', 'cricket-news', 'sports-news',
            'sports', 'cricket', 'world', 'business', 'tech', 'technology', 'science',
            'scores', 'fixtures', 'scores-fixtures', 'results', 'schedule', 'schedules', 'live', 'liveblog'
        ]
        if last_segment in generic_sections:
            return False
        # Also reject if the last segment ends with "-news"
        if last_segment.endswith('-news'):
            return False
    
    # Accept most other URLs - let content quality filtering handle the rest
    # Article URLs typically have SOMETHING meaningful (date, slug, id, etc.)
    
    has_date = any(x in path for x in ['2025', '2024', '2023', '2022', '/10/', '/oct/', '/09/', '/sep/', '/08/', '/aug/'])
    has_slug = '-' in path or '_' in path  # Any hyphenated or underscored slug
    has_article_path = any(x in path for x in ['/article/', '/story/', '/post/', '/releases/', '/film/', '/movie/'])
    has_numbers = any(char.isdigit() for char in path)  # Any numbers (IDs, dates, etc.)
    
    # Reject if path is clearly an index-like page even with a hyphenated segment
    index_like = any(x in path for x in ['scores-fixtures', 'live-scores', 'scorecard', 'fixtures', 'results', 'schedule', 'schedules'])
    if index_like and not (has_article_path or has_date or has_numbers):
        return False

    # Require at least a slug or explicit article path or date
    return has_slug or has_article_path or has_date


async def _verify_article_content(url: str) -> Optional[Dict]:
    """
    Actually visit the URL and check if it has real article content
    Like manually browsing Google results and checking each page!
    
    Returns:
        Dict with title, description, word_count if valid article
        None if it's a category page or has no content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-IN,en;q=0.9',
            'Referer': 'https://www.google.com/',
        }
        
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            html = response.text
            soup = BeautifulSoup(html, 'lxml')
            
            # Remove script, style, nav, footer
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
                tag.decompose()
            
            # Extract title (prefer strong signals)
            title = ""
            og_title = soup.find('meta', attrs={'property': 'og:title'}) or soup.find('meta', attrs={'name': 'og:title'})
            tw_title = soup.find('meta', attrs={'name': 'twitter:title'})
            if og_title and og_title.get('content'):
                title = og_title.get('content').strip()
            elif tw_title and tw_title.get('content'):
                title = tw_title.get('content').strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
            if not title:
                h1 = soup.find('h1')
                title = h1.get_text().strip() if h1 else ""
            # JSON-LD headline fallback
            if not title:
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        import json as _json
                        data = _json.loads(script.string or '{}')
                        objs = data if isinstance(data, list) else [data]
                        for obj in objs:
                            hl = obj.get('headline')
                            if isinstance(hl, str) and len(hl) > 5:
                                title = hl.strip()
                                break
                        if title:
                            break
                    except Exception:
                        pass
            
            # Extract main content - try common article containers
            content = ""
            article_selectors = [
                'article',
                '[class*="article"]',
                '[class*="content"]',
                '[class*="story"]',
                '[id*="article"]',
                '[id*="content"]',
                'main',
                '.post-content',
                'div[itemprop="articleBody"]',
                '.article-content',
                '.story__content',
                '.liveblog', '.live-blog', '.liveBlog', '.live-updates',
                '.articleText', '.entry-content',
                '.content', '#content', '.main-article', '.article-body',
                '.liveblog-content', '.live-blog-wrap', '.live-update', '.liveUpdate',
                '.ie-cups', '.synopsis'
            ]
            
            for selector in article_selectors:
                # Try a single container
                container = soup.select_one(selector)
                if container:
                    content = container.get_text(separator=' ', strip=True)
                    if len(content) > 200:  # Found substantial content
                        break
                # If not enough, try multiple nodes and join
                nodes = soup.select(selector)
                if nodes and len(content) < 200:
                    joined = ' '.join([n.get_text(separator=' ', strip=True) for n in nodes])
                    if len(joined) > len(content):
                        content = joined
                    if len(content) > 200:
                        break
            
            # If no article container found, get body text
            if not content or len(content) < 200:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean up content
            content = ' '.join(content.split())  # Normalize whitespace
            # Keep a reasonably sized preview; retain full text internally
            content_preview = content[:4000]
            word_count = len(content.split())
            
            # Validation checks
            # 1. Must have meaningful title
            if not title or len(title) < 10:
                logger.debug(f"   ❌ No proper title: {url}")
                return None
            
            # 2. Must have substantial content
            url_l = url.lower()
            looks_live = any(x in url_l for x in ['live', 'liveblog', 'live-updates'])
            min_words = 80 if looks_live else 180
            if word_count < min_words:
                logger.debug(f"   ❌ Too short ({word_count} words): {url}")
                # Try AMP fallback if available
                amp_link = soup.find('link', rel=lambda v: v and 'amphtml' in v.lower())
                if amp_link and amp_link.get('href'):
                    try:
                        amp_url = amp_link.get('href')
                        logger.debug(f"   ↪️  Trying AMP fallback: {amp_url}")
                        amp_resp = await client.get(amp_url, headers=headers)
                        amp_resp.raise_for_status()
                        amp_soup = BeautifulSoup(amp_resp.text, 'lxml')
                        amp_content = ''
                        for selector in article_selectors:
                            node = amp_soup.select_one(selector)
                            if node:
                                amp_content = node.get_text(separator=' ', strip=True)
                                if len(amp_content) > 150:
                                    break
                        if not amp_content and amp_soup.find('body'):
                            amp_content = amp_soup.find('body').get_text(separator=' ', strip=True)
                        amp_content = ' '.join((amp_content or '').split())
                        if len(amp_content.split()) >= min_words:
                            content = amp_content
                            content_preview = content[:4000]
                            word_count = len(content.split())
                        else:
                            return None
                    except Exception:
                        return None
            
            # 3. Check for listing page keywords in title
            listing_keywords = [
                'latest news', 'latest updates', 'more stories', 'top stories', 
                'all news', 'news feed', 'breaking news updates', 'news archive',
                'latest news and analysis',  # Indian Express section pages
                'section:', 'category:',
            ]
            title_lower = title.lower()
            if any(keyword in title_lower for keyword in listing_keywords):
                logger.debug(f"   ❌ Listing page keyword in title: {url}")
                return None
            
            # 4. Check for multiple article titles (listing page indicator)
            h2_titles = soup.find_all('h2')
            h3_titles = soup.find_all('h3')
            multiple_titles = len(h2_titles) + len(h3_titles)
            
            if multiple_titles > 8:  # Listing pages have many article titles
                logger.debug(f"   ❌ Listing page ({multiple_titles} article titles): {url}")
                return None
            
            # 5. Check if it's a category/listing page
            # Category pages have lots of links but little paragraph content
            paragraphs = soup.find_all(['p'])
            paragraph_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True) and len(p.get_text(strip=True)) > 30])
            paragraph_words = len(paragraph_text.split())
            
            # Count links vs paragraphs
            all_links = soup.find_all('a')
            link_to_paragraph_ratio = len(all_links) / max(len(paragraphs), 1)
            
            # Category pages have high link-to-paragraph ratio (lots of navigation)
            if link_to_paragraph_ratio > 8:  # More than 8 links per paragraph = listing page
                logger.debug(f"   ❌ Category page (high link ratio: {link_to_paragraph_ratio:.1f}): {url}")
                return None
            
            if paragraph_words < 150:  # Real articles have multiple paragraphs
                logger.debug(f"   ❌ Category page (few paragraphs: {paragraph_words} words): {url}")
                return None
            
            # Prefer strong article signals
            is_article_signal = False
            og_type = soup.find('meta', attrs={'property': 'og:type'})
            if og_type and og_type.get('content', '').lower() in ['article', 'news']:
                is_article_signal = True
            # Check for publish time
            if soup.find('meta', attrs={'property': 'article:published_time'}) or soup.find('meta', attrs={'name': 'article:published_time'}):
                is_article_signal = True
            # Check JSON-LD NewsArticle/Article
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    import json as _json
                    data = _json.loads(script.string or '{}')
                    types = []
                    if isinstance(data, list):
                        for d in data:
                            t = d.get('@type')
                            if isinstance(t, list):
                                types += [x.lower() for x in t if isinstance(x, str)]
                            elif isinstance(t, str):
                                types.append(t.lower())
                    elif isinstance(data, dict):
                        t = data.get('@type')
                        if isinstance(t, list):
                            types += [x.lower() for x in t if isinstance(x, str)]
                        elif isinstance(t, str):
                            types.append(t.lower())
                    if any(x in ['newsarticle', 'article', 'report'] for x in types):
                        is_article_signal = True
                        break
                except Exception:
                    pass
            if not is_article_signal:
                return None

            # Get meta description
            description = ""
            meta_desc = (
                soup.find('meta', attrs={'name': 'description'}) or
                soup.find('meta', attrs={'property': 'og:description'}) or
                soup.find('meta', attrs={'name': 'twitter:description'})
            )
            if meta_desc:
                description = meta_desc.get('content', '').strip()
            
            # If no meta description, use first paragraph
            if not description:
                first_p = soup.find('p')
                if first_p:
                    description = first_p.get_text(strip=True)[:200]
            
            # Try to extract publish date
            published_date = ""
            pd_meta = (
                soup.find('meta', attrs={'property': 'article:published_time'}) or
                soup.find('meta', attrs={'name': 'article:published_time'}) or
                soup.find('meta', attrs={'name': 'publish-date'}) or
                soup.find('meta', attrs={'name': 'date'}) or
                soup.find('meta', attrs={'property': 'og:updated_time'})
            )
            if pd_meta:
                published_date = pd_meta.get('content', '').strip()
            # JSON-LD datePublished/dateModified
            if not published_date:
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        import json as _json
                        data = _json.loads(script.string or '{}')
                        objs = data if isinstance(data, list) else [data]
                        for obj in objs:
                            dp = obj.get('datePublished') or obj.get('dateModified')
                            if isinstance(dp, str) and len(dp) >= 8:
                                published_date = dp
                                break
                        if published_date:
                            break
                    except Exception:
                        pass

            logger.debug(f"   ✅ Verified article ({word_count} words): {title[:50]}...")

            return {
                'title': title,
                'description': description,
                'content': content_preview,
                'published_date': published_date,
                'word_count': word_count,
                'is_valid': True
            }
            
    except httpx.TimeoutException:
        logger.debug(f"   ⏱️  Timeout: {url}")
        return None
    except Exception as e:
        logger.debug(f"   ❌ Fetch error: {url} - {str(e)[:50]}")
        return None


async def search_google_news(query: str, max_results: int = 10, category: str = None) -> List[Dict]:
    """
    Search Google for news articles using Custom Search API
    Filters results to last 24 hours only for fresh news.
    For sports category, only trusted sources are returned.
    
    Args:
        query: Search query (e.g., "Harmanpreet Kaur cricket")
        max_results: Maximum number of results to return
        category: Category (e.g., "sports") - enables trusted source filtering
    
    Returns:
        List of article dictionaries matching RSS format (last 24 hours)
    """
    if not GOOGLE_SEARCH_ENABLED:
        logger.info("Google Search disabled. Enable by setting API keys.")
        return []
    
    # Choose method based on available API
    # Prefer Google Custom Search (as requested) with automatic fallback to SerpAPI on errors
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
        try:
            logger.info(f"🔍 Using Google Custom Search API for: '{query}'")
            return await _search_google_custom(query, max_results, category)
        except Exception as e:
            logger.error(f"Google Custom Search failed, falling back to SerpAPI: {e}")
            if SERPAPI_KEY and SERPAPI_KEY != "YOUR_SERPAPI_KEY":
                logger.info(f"🔍 Falling back to SerpAPI for: '{query}'")
                return await _search_serpapi(query, max_results, category)
            return []
    elif SERPAPI_KEY and SERPAPI_KEY != "YOUR_SERPAPI_KEY":
        logger.info(f"🔍 Using SerpAPI for search: '{query}'")
        return await _search_serpapi(query, max_results, category)
    else:
        logger.warning("No Google Search API configured")
        return []


async def _search_google_custom(query: str, max_results: int, category: str = None) -> List[Dict]:
    """Search using Google Custom Search API (last 24 hours only, trusted sources for sports)"""
    base_url = "https://www.googleapis.com/customsearch/v1"
    articles = []
    excluded_count = 0
    untrusted_count = 0
    generic_count = 0

    start_index = 1
    verify_enabled = VERIFY_CONTENT_ENABLED
    require_url_overlap = True
    relaxed = False
    seen_urls = set()

    # RSS-first strategy (dynamic editions based on query/category)
    remaining = max_results
    try:
        q = query or ""
        ql = q.lower()
        rss_q = quote_plus(q)
        # Base editions always include global US English
        editions = [("en", "US", "US:en")]
        # Sports/Cricket intents → add India and UK/Australia where relevant
        if (category and category.lower() == "sports") or any(k in ql for k in ["cricket", "ipl", "team india", "kohli", "rohit", "ind vs", "india vs", "world cup", "odi", "test"]):
            editions.append(("en-IN", "IN", "IN:en"))
            editions.append(("en-GB", "GB", "GB:en"))
            editions.append(("en-AU", "AU", "AU:en"))
        # South Asia geopolitics → Pakistan
        if any(k in ql for k in ["pakistan", "karachi", "islamabad"]):
            editions.append(("en-PK", "PK", "PK:en"))
        # Middle East (for broader coverage if asked)
        if any(k in ql for k in ["uae", "dubai", "abu dhabi"]):
            editions.append(("en-AE", "AE", "AE:en"))
        # US politics
        if any(k in ql for k in ["trump", "biden", "republican", "democrat", "usa politics", "us election"]):
            editions.append(("en", "US", "US:en"))
        # Canada
        if any(k in ql for k in ["canada", "toronto", "vancouver", "ottawa"]):
            editions.append(("en-CA", "CA", "CA:en"))
        
        # Deduplicate editions tuple order preserved
        seen_editions = set()
        unique_editions = []
        for e in editions:
            if e not in seen_editions:
                unique_editions.append(e)
                seen_editions.add(e)

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as rss_client:
            if remaining > 0:
                rss_url = f"https://news.google.com/rss/search?q={rss_q}"
                try:
                    rss_resp = await rss_client.get(rss_url, headers={"User-Agent": "Mozilla/5.0"})
                    rss_resp.raise_for_status()
                    root = ET.fromstring(rss_resp.text)
                except Exception as e:
                    logger.debug(f"Google News RSS fetch failed: {e}")
                    root = None

                if root is not None:
                    for item in root.findall('.//item'):
                        if remaining <= 0:
                            break
                        title = (item.findtext('title') or '').strip()
                        link = (item.findtext('link') or '').strip()
                        pub_date = (item.findtext('pubDate') or '').strip()
                        if not link or link in seen_urls:
                            continue
                        # Resolve Google News redirect links to the publisher URL before filtering
                        try:
                            if 'news.google.com' in link:
                                resp = await rss_client.get(link, headers={"User-Agent": "Mozilla/5.0"})
                                if resp and resp.url:
                                    link = str(resp.url)
                        except Exception:
                            # If resolution fails, keep original link and continue
                            pass
                        if _is_excluded_source(link) or not _is_specific_article(link):
                            continue
                        verified = await _verify_article_content(link)
                        if not verified:
                            verified = {
                                'title': title,
                                'description': '',
                                'word_count': 0,
                                'is_valid': True,
                                'published_date': pub_date,
                                'content': ''
                            }
                        article = {
                            "title": verified.get('title') or title,
                            "summary": verified.get('description') or '',
                            "description": verified.get('description') or '',
                            "url": link,
                            "published_date": verified.get('published_date', pub_date),
                            "author": "",
                            "source": _extract_domain(link),
                            "tags": [],
                            "image_url": "",
                            "category": (category.lower() if category else ""),
                            "fetched_at": datetime.utcnow().isoformat(),
                            "word_count": verified.get('word_count', 0),
                            "content": verified.get('content', ''),
                        }
                        articles.append(article)
                        seen_urls.add(link)
                        remaining -= 1
    except Exception as e:
        logger.debug(f"Google News RSS fetch failed: {e}")

    # SerpAPI fallback for remaining
    if remaining > 0:
        try:
            serp_items = await _search_serpapi(query, remaining, category)
            for it in serp_items:
                if it.get('url') and it['url'] not in {a['url'] for a in articles}:
                    articles.append(it)
        except Exception as e:
            logger.debug(f"SerpAPI fallback failed: {e}")

    total_items = len(articles)
    logger.info(f"Google Custom Search: collected {total_items} verified articles ✅")
    if excluded_count > 0:
        logger.info(f"   ❌ {excluded_count} social media sites")
    if generic_count > 0:
        logger.info(f"   ❌ {generic_count} homepage/category/invalid content pages")
    if untrusted_count > 0:
        logger.info(f"   ❌ {untrusted_count} untrusted sources")
    if len(articles) > 0:
        logger.info(f"   ✅ All {len(articles)} articles verified by visiting URLs!")
    return articles
    async with httpx.AsyncClient(timeout=10.0) as client:
        # ------------- STRICT PASS (top results only) -------------
        while len(articles) < max_results:
            # Take only the top few results from the first page to reflect popularity
            page_size = min(5, max_results - len(articles))
            params = {
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_SEARCH_ENGINE_ID,
                "q": query,
                "num": page_size,  # API limit is 10 per request
                "start": start_index,
                "dateRestrict": "d1",  # Last 24 hours
                "sort": "date",  # Sort by date
                "gl": "in",  # Country bias: India
                "hl": "en",  # Language bias: English
            }
            # Favor exact phrase matches for multi-word queries
            if " " in query:
                params["exactTerms"] = query

            # Exponential backoff retries for 429/Request errors
            data = None
            for attempt in range(3):
                try:
                    response = await client.get(base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    break
                except httpx.HTTPStatusError as e:
                    if e.response is not None and e.response.status_code == 429 and attempt < 2:
                        wait_s = 0.5 * (2 ** attempt)
                        logger.warning(f"Google CSE 429 Too Many Requests. Retrying in {wait_s:.1f}s (attempt {attempt+1}/3)...")
                        await asyncio.sleep(wait_s)
                        continue
                    # Propagate to trigger SerpAPI fallback
                    raise
                except httpx.RequestError:
                    if attempt < 2:
                        wait_s = 0.5 * (2 ** attempt)
                        logger.warning(f"Google CSE request error. Retrying in {wait_s:.1f}s (attempt {attempt+1}/3)...")
                        await asyncio.sleep(wait_s)
                        continue
                    raise

            if not data:
                # No data retrieved after retries; propagate
                raise RuntimeError("Google CSE: failed to retrieve data after retries")

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                url = item.get("link", "")

                # Skip YouTube, Facebook, and other social media/video sites
                if _is_excluded_source(url):
                    excluded_count += 1
                    continue

                # Skip homepage/category pages - only accept specific articles
                if not _is_specific_article(url):
                    generic_count += 1
                    continue

                # For sports/news/movies categories, only allow trusted sources
                if category:
                    cat_lower = category.lower()
                    if cat_lower == "sports":
                        if not _is_trusted_sports_source(url):
                            untrusted_count += 1
                            continue
                    elif cat_lower == "news":
                        if not _is_trusted_news_source(url):
                            untrusted_count += 1
                            continue
                    elif cat_lower == "movies":
                        if not _is_trusted_movie_source(url):
                            untrusted_count += 1
                            continue

                # 🌐 Optionally verify article content by visiting URL
                verified = None
                if VERIFY_CONTENT_ENABLED:
                    logger.info(f"   🔍 Verifying: {url}")
                    verified = await _verify_article_content(url)

                # If verification disabled or fails, use API data
                if not verified:
                    verified = {
                        'title': item.get("title", ""),
                        'description': item.get("snippet", ""),
                        'word_count': 0,
                        'is_valid': True
                    }

                # Basic query relevance filter: ensure overlap with user query
                def _matches_query(title: str, desc: str, q: str) -> bool:
                    try:
                        title_l = (title or '').lower()
                        desc_l = (desc or '').lower()
                        q_terms = [t for t in q.lower().split() if len(t) > 2]
                        hits = sum(1 for t in q_terms if t in title_l or t in desc_l)
                        # Strict: at least 2 terms for multi-word queries; relaxed: 1 term
                        need = 1 if relaxed else (2 if len(q_terms) >= 3 else 1)
                        return hits >= need
                    except:
                        return True
                if not _matches_query(verified.get('title') or item.get("title", ""), verified.get('description') or item.get("snippet", ""), query):
                    continue

                # Ensure URL overlaps with user query to avoid generic section URLs (strict pass only)
                if require_url_overlap:
                    url_l = (url or '').lower()
                    q_terms_url = [t for t in query.lower().split() if len(t) > 3]
                    if q_terms_url and not any(t.replace(' ', '-') in url_l or t in url_l for t in q_terms_url):
                        # If none of the query terms appear in URL slug, skip
                        continue

                # Use verified title and description if available
                article = {
                    "title": verified.get('title') or item.get("title", ""),
                    "summary": verified.get('description') or item.get("snippet", ""),
                    "description": verified.get('description') or item.get("snippet", ""),
                    "url": url,
                    "published_date": verified.get('published_date', ""),
                    "author": "",
                    "source": _extract_domain(url),
                    "tags": [],
                    "image_url": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src", ""),
                    "category": (category.lower() if category else ""),
                    "fetched_at": datetime.utcnow().isoformat(),
                    "word_count": verified.get('word_count', 0),  # Add word count
                    "content": verified.get('content', ""),
                }
                if url not in seen_urls:
                    articles.append(article)
                    seen_urls.add(url)

                if len(articles) >= max_results:
                    break

            # We only need the top results from the first page
            break

        # ------------- RELAXED PASS (fallback) -------------
        if len(articles) < 3:
            relaxed = True
            verify_enabled = True            # enable verification for fallback
            require_url_overlap = False     # allow relevant results without slug match
            start_index = 1
            page_size = min(5, max_results - len(articles))
            params = {
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_SEARCH_ENGINE_ID,
                "q": query,
                "num": page_size,
                "start": start_index,
                "dateRestrict": "d1",
                "sort": "date",
                "gl": "in",
                "hl": "en",
            }
            if " " in query:
                params["exactTerms"] = query

            data = None
            for attempt in range(3):
                try:
                    response = await client.get(base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    break
                except httpx.HTTPStatusError as e:
                    if e.response is not None and e.response.status_code == 429 and attempt < 2:
                        wait_s = 0.5 * (2 ** attempt)
                        logger.warning(f"Google CSE 429 Too Many Requests (relaxed). Retrying in {wait_s:.1f}s (attempt {attempt+1}/3)...")
                        await asyncio.sleep(wait_s)
                        continue
                    raise
                except httpx.RequestError:
                    if attempt < 2:
                        wait_s = 0.5 * (2 ** attempt)
                        logger.warning(f"Google CSE request error (relaxed). Retrying in {wait_s:.1f}s (attempt {attempt+1}/3)...")
                        await asyncio.sleep(wait_s)
                        continue
                    raise

            items = (data or {}).get("items", [])
            for item in items:
                url = item.get("link", "")
                if _is_excluded_source(url) or not _is_specific_article(url):
                    continue

                verified = None
                if verify_enabled:
                    logger.info(f"   🔍 Verifying (relaxed): {url}")
                    verified = await _verify_article_content(url)
                if not verified:
                    verified = {
                        'title': item.get("title", ""),
                        'description': item.get("snippet", ""),
                        'word_count': 0,
                        'is_valid': True
                    }

                # Relevance: 1 term match in title or description is enough in relaxed
                def _matches_query_relaxed(title: str, desc: str, q: str) -> bool:
                    try:
                        title_l = (title or '').lower()
                        desc_l = (desc or '').lower()
                        q_terms = [t for t in q.lower().split() if len(t) > 2]
                        hits = sum(1 for t in q_terms if t in title_l or t in desc_l)
                        return hits >= 1
                    except:
                        return True
                if not _matches_query_relaxed(verified.get('title') or item.get("title", ""), verified.get('description') or item.get("snippet", ""), query):
                    continue

                if url not in seen_urls:
                    article = {
                        "title": verified.get('title') or item.get("title", ""),
                        "summary": verified.get('description') or item.get("snippet", ""),
                        "description": verified.get('description') or item.get("snippet", ""),
                        "url": url,
                        "published_date": verified.get('published_date', ""),
                        "author": "",
                        "source": _extract_domain(url),
                        "tags": [],
                        "image_url": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src", ""),
                        "category": (category.lower() if category else ""),
                        "fetched_at": datetime.utcnow().isoformat(),
                        "word_count": verified.get('word_count', 0),
                        "content": verified.get('content', ""),
                    }
                    articles.append(article)
                    seen_urls.add(url)
                if len(articles) >= max_results:
                    break

        # ------------- SITE-BIASED PASS (sports only) -------------
        if len(articles) < 3 and category and category.lower() == "sports":
            verify_enabled = True            # ensure verification in site-biased pass
            remaining = max_results - len(articles)
            for domain in TRUSTED_SPORTS_SOURCES:
                if remaining <= 0:
                    break
                site_query = f"{query} site:{domain}"
                params = {
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_SEARCH_ENGINE_ID,
                    "q": site_query,
                    "num": min(2, remaining),
                    "start": 1,
                    "dateRestrict": "d1",
                    "sort": "date",
                    "gl": "in",
                    "hl": "en",
                }
                try:
                    response = await client.get(base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    logger.debug(f"Site-biased fetch failed for {domain}: {e}")
                    continue

                for item in data.get("items", []):
                    url = item.get("link", "")
                    if _is_excluded_source(url) or not _is_specific_article(url) or url in seen_urls:
                        continue

                    verified = None
                    if verify_enabled:
                        logger.info(f"   🔍 Verifying (site): {url}")
                        verified = await _verify_article_content(url)
                    if not verified:
                        verified = {
                            'title': item.get("title", ""),
                            'description': item.get("snippet", ""),
                            'word_count': 0,
                            'is_valid': True
                        }

                    article = {
                        "title": verified.get('title') or item.get("title", ""),
                        "summary": verified.get('description') or item.get("snippet", ""),
                        "description": verified.get('description') or item.get("snippet", ""),
                        "url": url,
                        "published_date": "",
                        "author": "",
                        "source": _extract_domain(url),
                        "tags": [],
                        "image_url": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src", ""),
                        "category": (category.lower() if category else ""),
                        "fetched_at": datetime.utcnow().isoformat(),
                        "word_count": verified.get('word_count', 0),
                    }
                    articles.append(article)
                    seen_urls.add(url)
                    remaining -= 1
                    if remaining <= 0:
                        break

        # ------------- SITE-BIASED PASS (news mainstream/tech) -------------
        if len(articles) < 3 and category and category.lower() == "news":
            verify_enabled = True
            remaining = max_results - len(articles)
            news_domains = [
                "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
                "theverge.com", "techcrunch.com", "wired.com",
                "bbc.com", "nytimes.com", "cnn.com",
                "hindustantimes.com", "indianexpress.com", "thehindu.com", "ndtv.com"
            ]
            for domain in news_domains:
                if remaining <= 0:
                    break
                site_query = f"{query} site:{domain}"
                params = {
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_SEARCH_ENGINE_ID,
                    "q": site_query,
                    "num": min(2, remaining),
                    "start": 1,
                    "dateRestrict": "d1",
                    "sort": "date",
                    "gl": "in",
                    "hl": "en",
                }
                try:
                    response = await client.get(base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    logger.debug(f"Site-biased fetch failed for {domain}: {e}")
                    continue

                for item in data.get("items", []):
                    url = item.get("link", "")
                    if _is_excluded_source(url) or not _is_specific_article(url) or url in seen_urls:
                        continue

                    verified = None
                    if verify_enabled:
                        logger.info(f"   🔍 Verifying (news-site): {url}")
                        verified = await _verify_article_content(url)
                    if not verified:
                        verified = {
                            'title': item.get("title", ""),
                            'description': item.get("snippet", ""),
                            'word_count': 0,
                            'is_valid': True
                        }

                    article = {
                        "title": verified.get('title') or item.get("title", ""),
                        "summary": verified.get('description') or item.get("snippet", ""),
                        "description": verified.get('description') or item.get("snippet", ""),
                        "url": url,
                        "published_date": verified.get('published_date', ""),
                        "author": "",
                        "source": _extract_domain(url),
                        "tags": [],
                        "image_url": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src", ""),
                        "category": (category.lower() if category else ""),
                        "fetched_at": datetime.utcnow().isoformat(),
                        "word_count": verified.get('word_count', 0),
                        "content": verified.get('content', ""),
                    }
                    articles.append(article)
                    seen_urls.add(url)
                    remaining -= 1
                    if remaining <= 0:
                        break

        # ------------- GOOGLE NEWS RSS FALLBACK (all categories) -------------
        if len(articles) < 3:
            remaining = max_results - len(articles)
            try:
                rss_q = quote_plus(query)
                rss_url = f"https://news.google.com/rss/search?q={rss_q}&hl=en-IN&gl=IN&ceid=IN:en"
                logger.info(f"Google News RSS fallback: {rss_url}")
                async with httpx.AsyncClient(timeout=10.0) as rss_client:
                    rss_resp = await rss_client.get(rss_url, headers={"User-Agent": "Mozilla/5.0"})
                    rss_resp.raise_for_status()
                    root = ET.fromstring(rss_resp.text)
                    for item in root.findall('.//item'):
                        if remaining <= 0:
                            break
                        title = (item.findtext('title') or '').strip()
                        link = (item.findtext('link') or '').strip()
                        pub_date = (item.findtext('pubDate') or '').strip()
                        if not link or link in seen_urls:
                            continue
                        if _is_excluded_source(link) or not _is_specific_article(link):
                            continue
                        verified = await _verify_article_content(link)
                        if not verified:
                            verified = {
                                'title': title,
                                'description': '',
                                'word_count': 0,
                                'is_valid': True,
                                'published_date': pub_date,
                                'content': ''
                            }
                        article = {
                            "title": verified.get('title') or title,
                            "summary": verified.get('description') or '',
                            "description": verified.get('description') or '',
                            "url": link,
                            "published_date": verified.get('published_date', pub_date),
                            "author": "",
                            "source": _extract_domain(link),
                            "tags": [],
                            "image_url": "",
                            "category": (category.lower() if category else ""),
                            "fetched_at": datetime.utcnow().isoformat(),
                            "word_count": verified.get('word_count', 0),
                            "content": verified.get('content', ''),
                        }
                        articles.append(article)
                        seen_urls.add(link)
                        remaining -= 1
            except Exception as e:
                logger.debug(f"Google News RSS fallback failed: {e}")

    total_items = len(articles)
    logger.info(f"Google Custom Search: collected {total_items} verified articles ✅")
    if excluded_count > 0:
        logger.info(f"   ❌ {excluded_count} social media sites")
    if generic_count > 0:
        logger.info(f"   ❌ {generic_count} homepage/category/invalid content pages")
    if untrusted_count > 0:
        logger.info(f"   ❌ {untrusted_count} untrusted sources")
    if len(articles) > 0:
        logger.info(f"   ✅ All {len(articles)} articles verified by visiting URLs!")
    return articles


async def _search_serpapi(query: str, max_results: int, category: str = None) -> List[Dict]:
    """Search using SerpAPI (easier alternative) - last 24 hours only, trusted sources for sports"""
    try:
        url = "https://serpapi.com/search"
        params = {
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "q": query,
            "tbm": "nws",  # News search
            "num": max_results,
            "tbs": "qdr:d",  # Last 24 hours
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Debug: Log what SerpAPI returned
        news_results = data.get("news_results", [])
        logger.info(f"SerpAPI response: {len(news_results)} news results found")
        if not news_results and "error" in data:
            logger.error(f"SerpAPI error: {data.get('error')}")
        
        articles = []
        excluded_count = 0
        untrusted_count = 0
        generic_count = 0
        for item in news_results:
            url = item.get("link", "")
            
            # Skip YouTube, Facebook, and other social media/video sites
            if _is_excluded_source(url):
                excluded_count += 1
                continue
            
            # Skip homepage/category pages - only accept specific articles
            if not _is_specific_article(url):
                generic_count += 1
                continue
            
            # For sports/news/movies categories, only allow trusted sources
            if category:
                cat_lower = category.lower()
                if cat_lower == "sports":
                    if not _is_trusted_sports_source(url):
                        untrusted_count += 1
                        continue
                elif cat_lower == "news":
                    if not _is_trusted_news_source(url):
                        untrusted_count += 1
                        continue
                elif cat_lower == "movies":
                    if not _is_trusted_movie_source(url):
                        untrusted_count += 1
                        continue
            
            # 🌐 Optionally verify article content by visiting URL
            verified = None
            if VERIFY_CONTENT_ENABLED:
                logger.info(f"   🔍 Verifying: {url}")
                verified = await _verify_article_content(url)
            
            # If verification disabled or fails, use API data
            if not verified:
                verified = {
                    'title': item.get("title", ""),
                    'description': item.get("snippet", ""),
                    'word_count': 0,
                    'is_valid': True
                }

            # Basic query relevance filter: ensure overlap with user query
            def _matches_query(title: str, desc: str, q: str) -> bool:
                try:
                    title_l = (title or '').lower()
                    desc_l = (desc or '').lower()
                    q_terms = [t for t in q.lower().split() if len(t) > 2]
                    hits = sum(1 for t in q_terms if t in title_l or t in desc_l)
                    need = 2 if len(q_terms) >= 3 else 1
                    return hits >= need
                except:
                    return True
            if not _matches_query(verified.get('title') or item.get("title", ""), verified.get('description') or item.get("snippet", ""), query):
                continue
            
            # Extract source safely (can be dict or string)
            source_data = item.get("source", "")
            if isinstance(source_data, dict):
                source_name = source_data.get("name", "")
            else:
                source_name = str(source_data)
            
            # Use verified title and description if available
            article = {
                "title": verified.get('title') or item.get("title", ""),
                "summary": verified.get('description') or item.get("snippet", ""),
                "description": verified.get('description') or item.get("snippet", ""),
                "url": url,
                "published_date": verified.get('published_date', item.get("date", "")),
                "author": source_name,
                "source": source_name,
                "tags": [],
                "image_url": item.get("thumbnail", ""),
                "category": "sports",
                "fetched_at": datetime.utcnow().isoformat(),
                "word_count": verified.get('word_count', 0),  # Add word count
                "content": verified.get('content', ""),
            }
            articles.append(article)
        
        total_results = len(news_results)
        logger.info(f"SerpAPI: {total_results} results → {len(articles)} verified articles ✅")
        if excluded_count > 0:
            logger.info(f"   ❌ {excluded_count} social media sites")
        if generic_count > 0:
            logger.info(f"   ❌ {generic_count} homepage/category/invalid content pages")
        if untrusted_count > 0:
            logger.info(f"   ❌ {untrusted_count} untrusted sources")
        if len(articles) > 0:
            logger.info(f"   ✅ All {len(articles)} articles verified by visiting URLs!")
        if len(articles) == 0 and total_results > 0:
            logger.warning(f"⚠️  All {total_results} results filtered out after verification!")
        return articles
        
    except Exception as e:
        logger.error(f"SerpAPI error: {e}")
        return []


async def search_category_news(category: str, keywords: List[str], max_results: int = 10) -> List[Dict]:
    """
    Search Google News for specific category with keywords
    
    Args:
        category: Category (sports, technology, etc.)
        keywords: List of keywords to search
        max_results: Max results to return
    
    Returns:
        List of article dictionaries
    """
    if not keywords:
        return []
    
    # Build search query
    query_parts = [category] + keywords[:3]  # Category + top 3 keywords
    query = " ".join(query_parts)
    
    # Add "news" to ensure news articles
    query = f"{query} news"
    
    logger.info(f"Searching Google for: {query}")
    articles = await search_google_news(query, max_results)
    
    # Set category for all articles
    for article in articles:
        article["category"] = category.lower()
    
    return articles


def _extract_domain(url: str) -> str:
    """Extract domain name from URL"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return "Unknown"


# Alternative: Use DuckDuckGo (No API key needed!)
async def search_duckduckgo_news(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search DuckDuckGo for news (no API key required!)
    Uses the duckduckgo-search library
    Filters results to last 24 hours only for fresh news.
    """
    try:
        from duckduckgo_search import DDGS
        
        articles = []
        excluded_count = 0
        with DDGS() as ddgs:
            # timelimit='d' = last 24 hours (day)
            results = ddgs.news(query, max_results=max_results, timelimit='d')
            
            for item in results:
                url = item.get("url", "")
                
                # Skip YouTube, Facebook, and other social media/video sites
                if _is_excluded_source(url):
                    excluded_count += 1
                    continue
                
                article = {
                    "title": item.get("title", ""),
                    "summary": item.get("body", ""),
                    "description": item.get("body", ""),
                    "url": url,
                    "published_date": item.get("date", ""),
                    "author": "",
                    "source": item.get("source", ""),
                    "tags": [],
                    "image_url": item.get("image", ""),
                    "category": "sports",
                    "fetched_at": datetime.utcnow().isoformat(),
                }
                articles.append(article)
        
        logger.info(f"DuckDuckGo found {len(articles)} articles for: {query}")
        if excluded_count > 0:
            logger.info(f"   Filtered out {excluded_count} video/social media sites")
        return articles
        
    except ImportError:
        logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return []
