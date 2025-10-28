"""
Main Personalized News Pipeline
Orchestrates the 3-stage LLM-based RAG system:
1. LLM Preference Parsing → alertsparse
2. News Retrieval (RSS/Google/SERP) → raw articles
3. LLM Filtering & Ranking → notification_queue
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pymongo.database import Database

from services.llm_preference_parser import LLMPreferenceParser
from services.news_fetcher_service import NewsFetcherService
from services.llm_article_ranker import LLMArticleRanker

logger = logging.getLogger(__name__)


class PersonalizedNewsPipeline:
    """
    Main pipeline orchestrating all 3 stages of personalized news delivery.
    """
    
    def __init__(
        self,
        db: Database,
        gemini_api_key: str,
        model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize the pipeline with all required services.
        
        Args:
            db: MongoDB database instance
            gemini_api_key: Google Gemini API key
            model: Gemini model to use
        """
        self.db = db
        
        # Initialize collections
        self.alertsparse_collection = db.get_collection("alertsparse")
        self.notification_queue_collection = db.get_collection("notification_queue")
        self.alerts_collection = db.get_collection("alerts")
        
        # Initialize Stage services
        self.preference_parser = LLMPreferenceParser(gemini_api_key, model)
        self.news_fetcher = NewsFetcherService()
        self.article_ranker = LLMArticleRanker(gemini_api_key, model)
        
        logger.info("✅ Personalized News Pipeline initialized")
    
    # ==================== STAGE 1: PREFERENCE PARSING ====================
    
    async def parse_and_store_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Parse user alert using LLM and store in alertsparse collection.
        
        Args:
            alert_data: Raw alert with category, sub_categories, followup_questions, custom_question
            
        Returns:
            Parsed alertsparse object
        """
        try:
            alert_id = alert_data.get("alert_id")
            logger.info(f"📝 Stage 1: Parsing alert {alert_id}")
            
            # Check if already parsed
            existing = await self.alertsparse_collection.find_one({"alert_id": alert_id})
            if existing:
                logger.info(f"Alert {alert_id} already parsed, using existing")
                return existing
            
            # Parse with LLM
            alertsparse = await self.preference_parser.parse_user_alert(alert_data)
            
            # Store in database
            await self.alertsparse_collection.update_one(
                {"alert_id": alert_id},
                {"$set": alertsparse},
                upsert=True
            )
            
            logger.info(f"✅ Stage 1 Complete: Alert {alert_id} parsed and stored")
            logger.debug(f"Canonical entities: {alertsparse.get('canonical_entities')}")
            logger.debug(f"Event conditions: {alertsparse.get('event_conditions')}")
            
            return alertsparse
            
        except Exception as e:
            logger.error(f"❌ Stage 1 Error: {str(e)}")
            raise
    
    # ==================== STAGE 2: NEWS RETRIEVAL ====================
    
    async def fetch_news_for_alert(
        self,
        alertsparse: Dict[str, Any],
        max_articles: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Fetch news from multiple sources using contextual_query.
        
        Args:
            alertsparse: Parsed alert from Stage 1
            max_articles: Maximum articles to fetch
            
        Returns:
            List of raw articles
        """
        try:
            alert_id = alertsparse.get("alert_id")
            logger.info(f"🔍 Stage 2: Fetching news for alert {alert_id}")
            
            # Fetch from multiple sources
            articles = await self.news_fetcher.fetch_news_for_alert(
                alertsparse,
                max_articles=max_articles
            )
            
            logger.info(f"✅ Stage 2 Complete: Fetched {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Stage 2 Error: {str(e)}")
            return []
    
    # ==================== STAGE 3: LLM FILTERING & RANKING ====================
    
    async def filter_and_rank_articles(
        self,
        alertsparse: Dict[str, Any],
        articles: List[Dict[str, Any]],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Stage 3: Filter and rank articles using LLM reasoning.
        
        Args:
            alertsparse: Parsed alert from Stage 1
            articles: Raw articles from Stage 2
            top_n: Number of top articles to return
            
        Returns:
            Dict with filtered_ranked_articles, counts
        """
        try:
            alert_id = alertsparse.get("alert_id")
            logger.info(f"⚡ Stage 3: Filtering {len(articles)} articles for alert {alert_id}")
            
            # Filter and rank with LLM
            result = await self.article_ranker.filter_and_rank(
                alertsparse,
                articles,
                top_n=top_n
            )
            
            logger.info(
                f"✅ Stage 3 Complete: {result['included_count']} included, "
                f"{result['excluded_count']} excluded"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Stage 3 Error: {str(e)}")
            return {
                "filtered_ranked_articles": [],
                "included_count": 0,
                "excluded_count": len(articles)
            }
    
    # ==================== FULL PIPELINE ====================
    
    async def process_alert_full_pipeline(
        self,
        alert_data: Dict[str, Any],
        store_to_queue: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete 3-stage pipeline for an alert.
        
        Args:
            alert_data: Raw alert data
            store_to_queue: Whether to store results in notification_queue
            
        Returns:
            Complete pipeline result
        """
        try:
            alert_id = alert_data.get("alert_id")
            logger.info(f"🚀 Starting full pipeline for alert {alert_id}")
            
            # Stage 1: Parse preferences
            alertsparse = await self.parse_and_store_alert(alert_data)
            
            # Stage 2: Fetch news
            articles = await self.fetch_news_for_alert(alertsparse, max_articles=50)
            
            if not articles:
                logger.warning(f"No articles fetched for alert {alert_id}")
                return {
                    "alert_id": alert_id,
                    "status": "no_articles",
                    "articles": []
                }
            
            # Stage 3: Filter and rank
            result = await self.filter_and_rank_articles(alertsparse, articles, top_n=5)
            
            # Prepare final result
            final_result = {
                "alert_id": alert_id,
                "user_id": alertsparse.get("user_id"),
                "status": "success",
                "articles": result["filtered_ranked_articles"],
                "stats": {
                    "total_fetched": len(articles),
                    "included": result["included_count"],
                    "excluded": result["excluded_count"]
                },
                "alertsparse_snapshot": alertsparse,
                "processed_at": datetime.utcnow()
            }
            
            # Store to notification queue
            if store_to_queue and result["filtered_ranked_articles"]:
                await self.store_to_notification_queue(final_result)
            
            logger.info(
                f"✅ Pipeline Complete for {alert_id}: "
                f"{len(result['filtered_ranked_articles'])} articles ready"
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline Error for alert {alert_data.get('alert_id')}: {str(e)}")
            return {
                "alert_id": alert_data.get("alert_id"),
                "status": "error",
                "error": str(e),
                "articles": []
            }
    
    async def store_to_notification_queue(self, pipeline_result: Dict[str, Any]):
        """
        Store pipeline results to notification_queue for sending.
        
        Args:
            pipeline_result: Result from full pipeline
        """
        try:
            alert_id = pipeline_result["alert_id"]
            user_id = pipeline_result["user_id"]
            articles = pipeline_result["articles"]
            
            # Create notification queue entries (one per article)
            for article in articles:
                notification = {
                    "alert_id": alert_id,
                    "user_id": user_id,
                    "article": article,
                    "status": "pending",
                    "created_at": datetime.utcnow(),
                    "scheduled_for": datetime.utcnow(),  # Send immediately
                    "attempts": 0,
                    "max_attempts": 3
                }
                
                await self.notification_queue_collection.insert_one(notification)
            
            logger.info(f"📬 Stored {len(articles)} notifications to queue for alert {alert_id}")
            
        except Exception as e:
            logger.error(f"Error storing to notification queue: {str(e)}")
    
    # ==================== BATCH PROCESSING ====================
    
    async def process_all_active_alerts(self):
        """
        Process all active alerts in the system.
        This should be called by the scheduler (cron job).
        """
        try:
            logger.info("🔄 Starting batch processing of all active alerts")
            
            # Get all active alerts
            alerts = await self.alerts_collection.find({"is_active": True}).to_list(None)
            
            if not alerts:
                logger.info("No active alerts to process")
                return
            
            logger.info(f"Found {len(alerts)} active alerts to process")
            
            # Process each alert
            results = []
            for alert in alerts:
                try:
                    result = await self.process_alert_full_pipeline(alert, store_to_queue=True)
                    results.append(result)
                    
                    # Small delay between alerts to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing alert {alert.get('alert_id')}: {str(e)}")
                    continue
            
            # Summary
            success_count = sum(1 for r in results if r.get("status") == "success")
            total_articles = sum(len(r.get("articles", [])) for r in results)
            
            logger.info(
                f"✅ Batch processing complete: {success_count}/{len(alerts)} alerts processed, "
                f"{total_articles} total articles queued"
            )
            
            return {
                "total_alerts": len(alerts),
                "successful": success_count,
                "total_articles": total_articles,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return None
    
    async def close(self):
        """Clean up resources."""
        await self.news_fetcher.close()
        logger.info("Pipeline resources closed")


# Example usage
if __name__ == "__main__":
    async def test():
        from motor.motor_asyncio import AsyncIOMotorClient
        
        # Initialize DB
        client = AsyncIOMotorClient("mongodb://localhost:27017/")
        db = client["naarad_test"]
        
        # Initialize pipeline
        pipeline = PersonalizedNewsPipeline(
            db=db,
            gemini_api_key="YOUR_API_KEY"
        )
        
        # Test alert
        alert = {
            "alert_id": "test_123",
            "user_id": "user_456",
            "main_category": "Sports",
            "sub_categories": ["Cricket"],
            "followup_questions": ["team india", "world cup"],
            "custom_question": "only give me my team win"
        }
        
        # Run pipeline
        result = await pipeline.process_alert_full_pipeline(alert)
        
        print(f"\n✅ Pipeline Result:")
        print(f"Status: {result['status']}")
        print(f"Articles found: {len(result.get('articles', []))}")
        
        for i, article in enumerate(result.get('articles', [])[:3], 1):
            print(f"\n{i}. {article.get('title')}")
            print(f"   Relevance: {article.get('relevance_score')}")
            print(f"   Reason: {article.get('reason')}")
        
        await pipeline.close()
    
    asyncio.run(test())
