import logging
from typing import List, Dict
from datetime import datetime

from .config import CATEGORY_RSS_FEEDS, GEMINI_API_KEY
import google.generativeai as genai

try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None

logger = logging.getLogger(__name__)

# Google Search integration (optional)
GOOGLE_SEARCH_ENABLED = True  # ✅ ENABLED - Augment RSS with Google Search results

# Configure Gemini for query generation
genai.configure(api_key=GEMINI_API_KEY)

# Log module load
logger.warning("🚀🚀🚀 RSS_FETCHER MODULE LOADED WITH GOOGLE SEARCH + GEMINI 🚀🚀🚀")


async def _generate_search_queries_with_gemini(keywords: List[str], category: str) -> List[str]:
    """
    🤖 Use Gemini AI to generate intelligent search queries from keywords
    
    Example:
        Input: ["gujarat", "cm"]
        Output: ["Gujarat Chief Minister latest news", 
                 "Gujarat CM government announcements",
                 "Gujarat state government updates"]
    """
    try:
        # Prepare prompt
        keywords_str = ", ".join(keywords[:5])  # Top 5 keywords
        
        prompt = f"""You are a search query expert. Generate 3 effective Google search queries for news articles.

Category: {category}
User keywords: {keywords_str}

Requirements:
1. Create natural, specific search queries (not just keyword combinations)
2. Each query should be 3-5 words
3. Focus on finding NEWS articles (not general info)
4. Make queries diverse to find different angles
5. Return ONLY the 3 queries, one per line, no numbering or explanation

Example:
Input: "delhi, pollution, air quality"
Output:
Delhi air pollution latest news
Delhi AQI levels today
Air quality crisis Delhi NCR

Now generate for: {keywords_str}
"""
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Use synchronous call for now (async might have issues)
        import asyncio
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 100,
            }
        )
        
        # Parse response
        queries = []
        if response and response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering if present (1., 2., etc.)
                line = line.lstrip('0123456789.- ')
                if line and len(line) > 5:
                    queries.append(line)
        
        if queries:
            logger.info(f"🤖 Gemini generated queries: {queries}")
            return queries[:3]  # Top 3
        else:
            logger.warning("⚠️  Gemini returned no queries, using fallback")
            return []
            
    except Exception as e:
        logger.error(f"❌ Gemini query generation failed: {e}")
        return []


def _coalesce(*vals):
    for v in vals:
        if v:
            return v
    return ""


def _entry_to_article(entry, category: str, source: str) -> Dict:
    title = _coalesce(getattr(entry, "title", None), entry.get("title"))
    summary = _coalesce(getattr(entry, "summary", None), entry.get("summary"), entry.get("description"))
    description = _coalesce(entry.get("description"), summary)
    link = _coalesce(getattr(entry, "link", None), entry.get("link"))
    published = _coalesce(getattr(entry, "published", None), entry.get("published"), entry.get("updated"))
    image = ""
    try:
        media_content = entry.get("media_content") or []
        if isinstance(media_content, list) and media_content:
            image = media_content[0].get("url", "")
    except Exception:
        image = ""
    return {
        "title": title or "",
        "summary": summary or "",
        "description": description or "",
        "url": link or "",
        "published_date": published or "",
        "author": entry.get("author", ""),
        "source": source,
        "tags": [t.get("term") for t in (entry.get("tags") or []) if isinstance(t, dict) and t.get("term")],
        "image_url": image,
        "category": category,
        "fetched_at": datetime.utcnow().isoformat(),
    }


async def fetch_category_articles(category: str, keywords: List[str] = None) -> List[Dict]:
    """
    Fetch RSS articles for a category using feedparser. 
    Optionally augment with Google Search results if enabled.
    
    Args:
        category: Category to fetch (sports, technology, etc.)
        keywords: Optional keywords to enhance Google Search
    
    Returns:
        List of article dictionaries
    """
    cat = (category or "").lower()
    logger.warning(f"🎬🎬🎬 FETCH_CATEGORY_ARTICLES CALLED: cat={cat}, keywords={keywords} 🎬🎬🎬")
    feeds = CATEGORY_RSS_FEEDS.get(cat, [])
    if not feeds:
        return []
    if feedparser is None:
        logger.warning("feedparser is not installed. Run: pip install feedparser")
        return []

    # Fetch RSS articles
    out: List[Dict] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            source = parsed.feed.get("title", "") if getattr(parsed, "feed", None) else ""
            for e in parsed.entries:
                article = _entry_to_article(e, cat, source)
                article["from_google_search"] = False  # Mark as RSS article
                out.append(article)
        except Exception as e:
            logger.warning(f"RSS parse failed for {url}: {e}")
    
    logger.info(f"📰 RSS fetched {len(out)} articles for category: {cat}")
    
    # Optionally augment with Google Search (if enabled)
    # Note: Even if keywords is empty, we can use category-specific fallback terms
    logger.warning(f"🔍 GOOGLE_SEARCH_ENABLED = {GOOGLE_SEARCH_ENABLED}")
    if GOOGLE_SEARCH_ENABLED:
        logger.warning(f"🚀🚀🚀 STARTING GOOGLE SEARCH AUGMENTATION 🚀🚀🚀")
        try:
            from .google_search import search_google_news
            
            # Search for EACH keyword individually to find specific articles
            # This finds articles RSS missed about specific players/topics
            all_google_articles = []
            
            # DYNAMIC KEYWORD EXTRACTION - Works for ANY user input!
            # Filter out non-topical keywords (format preferences, sources, etc.)
            skip_keywords = [
                "mainstream", "independent", "brief", "summaries", "paragraphs",
                "links", "full articles", "times", "day", "sources", "media",
                "recommendations", "based on", "industry news", "announcements",
                "spoilers", "theatrical", "streaming", "mix of both", "ok",
                "breaking news", "top stories", "analysis", "latest", "updates",
                "give me", "all", "news", "want", "need", "show"
            ]
            
            # Handle keywords safely (might be None or empty)
            keywords_list = keywords if keywords else []
            
            # Step 1: Extract ALL meaningful keywords from user input
            topical_keywords = []
            for kw in keywords_list:
                kw_str = str(kw).strip()
                kw_lower = kw_str.lower()
                
                # Skip if too short or in skip list
                if len(kw_str) < 2:
                    continue
                if any(skip in kw_lower for skip in skip_keywords):
                    continue
                
                topical_keywords.append(kw_str)
            
            # Step 2: Intelligent tag building for search queries
            # Automatically combine related keywords for better search
            search_queries = []
            gemini_queries = []  # Initialize for later reference
            
            if topical_keywords:
                logger.warning(f"🎯🎯🎯 EXTRACTED USER KEYWORDS: {topical_keywords} 🎯🎯🎯")
                
                # 🤖 Try Gemini AI to generate intelligent search queries (with timeout)
                try:
                    logger.info("🤖 Asking Gemini to generate smart search queries...")
                    gemini_queries = await _generate_search_queries_with_gemini(topical_keywords, cat)
                    
                    if gemini_queries:
                        # Use Gemini's intelligent queries
                        search_queries = gemini_queries
                        logger.info(f"✨ Using Gemini-generated queries: {search_queries}")
                except Exception as e:
                    logger.error(f"❌ Gemini query generation error: {e}")
                    gemini_queries = []
                
                # Fallback: Manual combination if Gemini fails or returns nothing
                if not search_queries:
                    logger.info("⚠️  Using manual query generation")
                    if len(topical_keywords) >= 2:
                        # Combine first 2 keywords (e.g., "gujarat" + "cm" = "gujarat cm")
                        combined = " ".join(topical_keywords[:2])
                        search_queries.append(combined)
                        logger.info(f"🔗 Combined query: '{combined}'")
                        
                        # For politics, also try with full term
                        if any("cm" in kw.lower() for kw in topical_keywords):
                            # Add "chief minister" variant
                            full_term = combined.replace(" cm", " chief minister")
                            search_queries.append(full_term)
                            logger.info(f"🔗 Expanded query: '{full_term}'")
                    
                    # Also search each keyword individually (top 2)
                    for kw in topical_keywords[:2]:
                        if kw not in search_queries:  # Avoid duplicates
                            search_queries.append(kw)
                    
                    # Ensure we have at least one query
                    if not search_queries and topical_keywords:
                        search_queries = topical_keywords[:3]
                        logger.info(f"📍 Using keywords as-is: {search_queries}")
                
            else:
                # No user keywords - use intelligent category defaults
                logger.info(f"  No topical keywords found, using smart defaults for {cat}")
                
                # Detect subcategory from keywords_list
                has_sci = any("sci" in str(kw).lower() or "bio" in str(kw).lower() or "science" in str(kw).lower() for kw in keywords_list)
                has_politics = any("politic" in str(kw).lower() or "government" in str(kw).lower() or "minister" in str(kw).lower() for kw in keywords_list)
                
                if cat == "news" and has_sci:
                    search_queries = ["science discovery", "biology research", "medical breakthrough"]
                    logger.info(" Using science defaults")
                elif cat == "news" and has_politics:
                    search_queries = ["government", "politics", "chief minister"]
                    logger.info(" Using politics defaults")
                elif cat == "news":
                    search_queries = ["breaking news", "top stories", "world news"]
                elif cat == "movies":
                    search_queries = ["movie reviews", "new releases", "box office"]
                elif cat == "sports":
                    search_queries = ["match results", "live scores", "tournament"]
                else:
                    search_queries = [cat]  # Just use category name
            
            # Use search_queries instead of topical_keywords
            topical_keywords = search_queries
            
            logger.warning(f"📝📝📝 FINAL SEARCH QUERIES: {topical_keywords[:3]} 📝📝📝")
            logger.warning(f"🔍🔍🔍 STARTING GOOGLE SEARCH FOR {len(topical_keywords[:3])} QUERIES 🔍🔍🔍")
            
            # Search top 3 queries individually
            for i, query in enumerate(topical_keywords[:3], 1):
                # Check if Gemini generated this query (already has context)
                # vs manual query (needs category added)
                if gemini_queries and query in gemini_queries:
                    # Gemini query - use as-is (already optimized)
                    search_query = query
                    logger.info(f"🔍 Query {i} (Gemini): '{search_query}'")
                else:
                    # Manual query - add category for context
                    search_query = f"{query} {cat}"
                    logger.info(f"🔍 Query {i} (Manual): '{search_query}'")
                
                # Get 5-7 articles per query
                query_articles = await search_google_news(search_query, max_results=7, category=cat)
                
                if query_articles:
                    logger.info(f"   ✅ Found {len(query_articles)} articles")
                    all_google_articles.extend(query_articles)
            
            # Deduplicate by URL (in case same article matches multiple keywords)
            seen_urls = set()
            unique_google_articles = []
            for article in all_google_articles:
                url = article.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_google_articles.append(article)
            
            if unique_google_articles:
                logger.info(f"✅ Google Search added {len(unique_google_articles)} unique articles")
                
                # Mark Google Search articles with a flag for tracking
                for article in unique_google_articles:
                    article["from_google_search"] = True
                
                # Log sample Google Search articles
                for i, art in enumerate(unique_google_articles[:3], 1):
                    logger.info(f"   {i}. {art.get('title', 'NO TITLE')[:60]}... (Source: {art.get('source', 'unknown')})")
                
                out.extend(unique_google_articles)
            
        except Exception as e:
            logger.warning(f"Google Search augmentation failed: {e}")
    
    return out
