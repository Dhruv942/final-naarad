#!/usr/bin/env python3
"""
Test news fetching directly
"""

import asyncio
from services.news_fetcher_service import NewsFetcherService


async def test_news_fetching():
    """Test if news fetching works"""
    
    print("\n" + "="*70)
    print("📰 TESTING NEWS FETCHING")
    print("="*70)
    
    fetcher = NewsFetcherService()
    
    # Test query
    query = "india cricket team win"
    
    print(f"\nQuery: {query}")
    print("\nTesting individual sources:\n")
    
    # Test 1: Google News
    print("1️⃣  Testing Google News RSS...")
    try:
        articles = await fetcher._fetch_google_news(query, "cricket")
        print(f"   ✅ Found {len(articles)} articles")
        if articles:
            print(f"   Sample: {articles[0]['title'][:60]}...")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 2: Official RSS
    print("\n2️⃣  Testing Official RSS Feeds...")
    try:
        articles = await fetcher._fetch_official_rss("cricket")
        print(f"   ✅ Found {len(articles)} articles")
        if articles:
            print(f"   Sample: {articles[0]['title'][:60]}...")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 3: SERP (DuckDuckGo)
    print("\n3️⃣  Testing SERP Search...")
    try:
        articles = await fetcher._fetch_serp_fallback(query)
        print(f"   ✅ Found {len(articles)} articles")
        if articles:
            print(f"   Sample: {articles[0]['title'][:60]}...")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "-"*70)
    print("📊 COMPLETE FETCH TEST")
    print("-"*70)
    
    alertsparse = {
        "contextual_query": query,
        "category": "cricket"
    }
    
    print(f"\nFetching with full service...")
    articles = await fetcher.fetch_news_for_alert(alertsparse, max_articles=10)
    
    print(f"\n✅ Total Articles: {len(articles)}")
    
    if articles:
        print("\n📄 Sample Articles:")
        for i, article in enumerate(articles[:5], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   URL: {article['url'][:60]}...")
    else:
        print("\n⚠️  No articles found!")
        print("\nPossible reasons:")
        print("1. RSS feeds are timing out")
        print("2. Network issues or firewall blocking")
        print("3. RSS feed URLs have changed")
        print("4. Need to install: pip install feedparser httpx duckduckgo-search")
    
    await fetcher.close()
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(test_news_fetching())
