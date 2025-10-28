"""
Configuration module for RAG News System
Contains API keys, constants, and RSS feed configurations
"""
import os
import json

# API Configuration
GEMINI_API_KEY = 'AIzaSyBJqkrD1MteQ9FV6v3Dtdo39dhLUf4BRB4'
GOOGLE_API_KEY = 'AIzaSyBmIbpuMr1eBvLgqgk5_U-Kr4I3JKvrX-s'
GOOGLE_CX = '17f37cf9b3d404f47'  # Google Custom Search Engine ID

# WATI WhatsApp Configuration
WATI_ACCESS_TOKEN = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJiZGVmNjQ0OS02NDU3LTRiNDYtOTM4Mi03YjNiYmRmMmY2NGIiLCJ1bmlxdWVfbmFtZSI6ImFjdHVhbGx5dXNlZnVsZXh0ZW5zaW9uc0BnbWFpbC5jb20iLCJuYW1laWQiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiZW1haWwiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMjgvMjAyNSAxMjowNzo1OCIsInRlbmFudF9pZCI6IjQ1ODkxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.WPoEwLq2UdUs8Rl61SklQMFQ699mj1CqQ2v7iPZunuU'
WATI_BASE_URL = 'https://live-mt-server.wati.io/458913'
WATI_TEMPLATE_NAME = 'sports'
WATI_BROADCAST_NAME = 'sports_290920250931'

# RSS Feeds Configuration
CATEGORY_RSS_FEEDS = {
    "sports": [
        "https://www.thehindu.com/sport/feeder/default.rss",
        "https://www.thehindu.com/sport/football/feeder/default.rss",
        "https://feeds.bbci.co.uk/sport/rss.xml",
        "https://rss.cnn.com/rss/edition_sport.rss"
    ],
    "news": [
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/reuters/topNews",
        # Politics Feeds (Indian & Global)
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://feeds.reuters.com/reuters/INpoliticsNews",
        # Science & Biology Feeds
        "https://www.thehindu.com/sci-tech/feeder/default.rss",
        "https://rss.cnn.com/rss/edition_space.rss",
        "https://feeds.reuters.com/reuters/scienceNews",
        "https://www.nature.com/nature.rss",
        "https://www.sciencedaily.com/rss/all.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"
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

# Default trusted sources; can be overridden via env at runtime
DEFAULT_TRUSTED_SPORT_SOURCES = {
    "cricket": {
        "trusted_source": "cricbuzz.com",
        "query_template": '"{entity}" "live" site:cricbuzz.com'
    },
    "football": {
        "trusted_source": "livescore.com",
        "query_template": '"{entity}" "live" site:livescore.com'
    },
    "tennis": {
        "trusted_source": "atptour.com",
        "query_template": '"{entity}" "live" site:atptour.com'
    },
    "basketball": {
        "trusted_source": "nba.com",
        "query_template": '"{entity}" "live" site:nba.com'
    }
}

def get_trusted_sport_sources() -> dict:
    """Return trusted sport sources, allowing dynamic overrides via env.

    Env variable TRUSTED_SPORT_SOURCES_JSON may contain a JSON object like:
    {
      "cricket": {"trusted_source": "cricbuzz.com", "query_template": "\"{entity}\" \"live\" site:cricbuzz.com"},
      "football": {"trusted_source": "livescore.com", "query_template": "..."}
    }
    """
    cfg = DEFAULT_TRUSTED_SPORT_SOURCES.copy()
    raw = os.getenv("TRUSTED_SPORT_SOURCES_JSON")
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                for k, v in override.items():
                    if isinstance(v, dict):
                        base = cfg.get(k.lower(), {})
                        base.update(v)
                        cfg[k.lower()] = base
        except Exception:
            # Ignore malformed env override
            pass
    return cfg

def build_trusted_query(sport: str, entity: str) -> str | None:
    if not sport or not entity:
        return None
    mapping = get_trusted_sport_sources()
    spec = mapping.get(sport.lower())
    if not spec:
        return None
    try:
        return spec["query_template"].format(entity=entity)
    except Exception:
        return None
