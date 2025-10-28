#!/usr/bin/env python3
"""
Complete test of the 3-stage RAG pipeline
"""

import asyncio
import sys
from core.rag_pipeline import RAGPipeline, LLMClient, NewsFetcher
from db.mongo import db

USER_ID = "7ef93f95-626d-4488-8732-7f39edd282e6"


async def test_full_pipeline():
    """Test the complete 3-stage pipeline"""
    
    print("\n" + "="*70)
    print("🧪 COMPLETE 3-STAGE PIPELINE TEST")
    print("="*70)
    
    # Test alert data
    alert_data = {
        "alert_id": "test-123",
        "user_id": USER_ID,
        "category": "Sports",
        "sub_categories": ["cricket"],
        "followup_questions": ["team india", "worldcup", "test", "ODI"],
        "custom_question": "only give me my team win"
    }
    
    print("\n📋 Alert Data:")
    for key, value in alert_data.items():
        print(f"   {key}: {value}")
    
    # Initialize components
    print("\n⚙️  Initializing Components...")
    llm_client = LLMClient()
    news_fetcher = NewsFetcher()
    pipeline = RAGPipeline(db, llm_client, news_fetcher)
    
    try:
        print("\n" + "-"*70)
        print("📍 STAGE 1: LLM Preference Parsing")
        print("-"*70)
        
        parsed = await pipeline._parse_alert_preferences(alert_data)
        
        print(f"\n✅ Parsed Alert:")
        print(f"   Category: {parsed.category}")
        print(f"   Sub-categories: {parsed.sub_categories}")
        print(f"   Custom Question: {parsed.custom_question}")
        print(f"\n🤖 LLM Extracted:")
        print(f"   Canonical Entities: {parsed.canonical_entities}")
        print(f"   Event Conditions: {parsed.event_conditions}")
        print(f"   Contextual Query: {parsed.contextual_query}")
        print(f"   Forbidden Topics: {parsed.forbidden_topics}")
        
        if not parsed.contextual_query:
            print("\n⚠️  WARNING: contextual_query is empty!")
            print("   This will affect article fetching in Stage 2")
        
        print("\n" + "-"*70)
        print("📍 STAGE 2: RAG Retrieval (News Fetching)")
        print("-"*70)
        
        print(f"\nQuery: {parsed.contextual_query or 'cricket team india win'}")
        articles = await pipeline._fetch_articles(parsed)
        
        print(f"\n✅ Fetched {len(articles)} articles")
        if articles:
            print("\nSample Articles:")
            for i, article in enumerate(articles[:5], 1):
                print(f"   {i}. {article.title[:60]}...")
                print(f"      Source: {article.source}")
        else:
            print("\n⚠️  No articles fetched!")
            print("   Possible reasons:")
            print("   1. Empty contextual_query from Stage 1")
            print("   2. Network issues")
            print("   3. RSS feeds not responding")
        
        print("\n" + "-"*70)
        print("📍 STAGE 3: LLM Filtering + Ranking")
        print("-"*70)
        
        if articles:
            results = await pipeline._filter_and_rank_articles(parsed, articles)
            
            filtered = results.get("filtered_ranked_articles", [])
            print(f"\n✅ Filtered: {len(filtered)} relevant articles")
            print(f"   Included: {results.get('included_count', 0)}")
            print(f"   Excluded: {results.get('excluded_count', 0)}")
            
            if filtered:
                print("\nTop Articles:")
                for i, article in enumerate(filtered[:3], 1):
                    print(f"   {i}. {article.title}")
                    print(f"      URL: {article.url}")
        else:
            print("\n⚠️  Skipping (no articles to filter)")
        
        print("\n" + "-"*70)
        print("📬 NOTIFICATION QUEUE")
        print("-"*70)
        
        if articles:
            result = await pipeline._store_results(parsed, results)
            
            if result.get('status') == 'no_articles':
                print("\n❌ No articles queued")
            else:
                print(f"\n✅ Queued {result.get('stats', {}).get('queued', 0)} articles")
                print(f"   Alert ID: {result.get('alert_id')}")
                print(f"   Status: {result.get('status')}")
        else:
            print("\n⚠️  No articles to queue")
        
        print("\n" + "="*70)
        print("✅ PIPELINE TEST COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await llm_client.close()
        if hasattr(news_fetcher, 'fetcher') and news_fetcher.fetcher:
            await news_fetcher.fetcher.close()


if __name__ == "__main__":
    print("\n🚀 Starting comprehensive pipeline test...")
    asyncio.run(test_full_pipeline())
