"""
Configuration module for RAG News System
Contains API keys, constants, and RSS feed configurations
"""

# API Configuration
GEMINI_API_KEY = 'AIzaSyBJqkrD1MteQ9FV6v3Dtdo39dhLUf4BRB4'

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

# Per-category query enrichment (data-driven synonyms/boost terms)
# These tokens are appended to contextual queries when the category matches.
CATEGORY_QUERY_SYNONYMS = {
    "sports": [
        "live scores",
        "match results",
        "fixtures",
        "squad updates",
    ],
    "technology": [
        "product launches",
        "startup funding",
        "ai research",
    ],
    "movies": [
        "trailers",
        "box office",
        "reviews",
    ],
    "news": [
        "breaking news",
        "top stories",
        "analysis",
    ],
    "politics": [
        "government",
        "parliament",
        "chief minister",
        "cm",
        "prime minister",
        "elections",
        "policy",
        "minister",
    ],
    "politicls": [  # Handle typo
        "government",
        "parliament",
        "chief minister",
        "cm",
        "prime minister",
        "elections",
        "policy",
        "minister",
    ],
    "sci_bio": [
        "research",
        "study",
        "discovery",
        "breakthrough",
        "scientists",
        "biology",
        "medicine",
        "health",
    ],
}

# Scheduler Configuration
SCHEDULER_INTERVAL_SECONDS = 1800  # 30 minutes

# AI Configuration
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_TOKENS = 1024

# Filtering Configuration
ARTICLE_BATCH_SIZE = 3
MAX_ARTICLES_PER_FEED = 20
MAX_RELEVANT_ARTICLES = 15
ARTICLE_TIME_WINDOW_HOURS = 48

# Relevance Score Thresholds
MIN_RELEVANCE_SCORE = 0.05
MIN_KEYWORD_SCORE = 0.15
MIN_SEMANTIC_SCORE = 0.1

# Personalization and Scoring Weights (tunable)
# Final score ~ (w_semantic*semantic + w_personal*personal + w_bm25*bm25 + w_popularity*popularity) * fresh + gender_bonus
W_SEMANTIC = 0.5
W_PERSONAL = 0.2
W_BM25 = 0.0  # enable later if BM25 added
W_POPULARITY = 0.3

# Popularity behavior
POPULARITY_THRESHOLD = 0.5  # used when popularity_mode == "only_popular"

# Freshness decay per category (in hours)
FRESH_TAU_BY_CATEGORY = {
    "sports": 24,
    "news": 24,
    "technology": 48,
    "movies": 72,
}

# Gender facet bonus (small nudge when matched)
GENDER_BONUS = 0.05

# Source authority weights (0-1). Default used if domain not listed.
SOURCE_AUTHORITY = {
    "bbc.co.uk": 1.0,
    "reuters.com": 0.95,
    "thehindu.com": 0.9,
    "cnn.com": 0.85,
    "edition.cnn.com": 0.85,
    "feeds.feedburner.com": 0.75,
    "variety.com": 0.8,
    "default": 0.7,
}

# Intent-driven bonuses/penalties (tunable)
FORMAT_MATCH_BONUS = 0.08            # when article format matches explicit followup (odi/t20/test)
FORMAT_MISMATCH_PENALTY = 0.08       # when article format conflicts with explicit followup
HIGHLIGHTS_BONUS = 0.05              # when "prefer highlights" and article looks like highlights
FINAL_SCORES_BONUS = 0.04            # when "include final scores" and article contains score patterns
NEGATIVE_PENALTY = 0.15              # when custom says no betting/gossip and such terms are present
EVENT_BONUS = 0.08                   # big-event phrases: win|won|champions|world cup|final|clinched

# Intent pattern configuration (data-driven)
# These regex fragments are used to detect preferences and article facets.
INTENT_PATTERNS = {
    "formats": {
        "odi": [r"\bodi\b", r"one\s*day\s*international"],
        "t20": [r"\bt20\b", r"twenty\s*20"],
        "test": [r"\btest\b"],
    },
    "positives": {
        "highlights": [r"highlights", r"watch\s+highlights", r"video\s+highlights"],
        "final_scores": [r"\b\d{2,3}\/\d{1,2}\b", r"\b\d+-\d+\b"],
    },
    "negatives": {
        "betting": [r"betting", r"odds"],
        "gossip": [r"gossip", r"rumors"],
    },
    "events": [r"\bwin\b", r"\bwon\b", r"champions", r"world\s+cup", r"\bfinal\b", r"clinched"],
    "intents": {
        # user preference phrases in followups/custom
        "prefer_highlights": [r"prefer\s+highlights"],
        "final_scores": [r"include\s+final\s+scores", r"final\s+scores"],
        "no_betting": [r"no\s+betting", r"no\s+betting\s+odds", r"no\s+odds"],
        "no_gossip": [r"no\s+gossip", r"no\s+rumors"],
        "prefer_women": [r"\bwomen'?s?\b", r"\bwpl\b"],
        "prefer_men": [r"\bmen'?s?\b"],
    },
    "genders": {
        "women": [r"\bwomen'?s?\b", r"\bwpl\b"],
        "men": [r"\bmen'?s?\b"],
    },
}

# Gender facet bonus when explicitly preferred by user
EXPLICIT_GENDER_BONUS = 0.06
