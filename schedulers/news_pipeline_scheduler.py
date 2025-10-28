"""
News Pipeline Scheduler
Runs the 3-stage personalized news pipeline periodically (every hour or configurable interval)
Processes all active alerts and queues notifications.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import os

from core.personalized_news_pipeline import PersonalizedNewsPipeline
from db.mongo import db

logger = logging.getLogger(__name__)


class NewsPipelineScheduler:
    """
    Scheduler for running the personalized news pipeline periodically.
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        interval_minutes: int = 60,
        model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize the scheduler.
        
        Args:
            gemini_api_key: Google Gemini API key
            interval_minutes: How often to run the pipeline (default: 60 minutes)
            model: Gemini model to use
        """
        self.interval_minutes = interval_minutes
        self.running = False
        self.last_run = None
        
        # Initialize pipeline
        self.pipeline = PersonalizedNewsPipeline(
            db=db,
            gemini_api_key=gemini_api_key,
            model=model
        )
        
        logger.info(
            f"📅 News Pipeline Scheduler initialized "
            f"(interval: {interval_minutes} minutes)"
        )
    
    async def start(self):
        """Start the scheduler loop."""
        self.running = True
        logger.info("🚀 News Pipeline Scheduler started")
        
        while self.running:
            try:
                await self._run_pipeline_cycle()
                
                # Wait for next cycle
                logger.info(f"⏰ Next run in {self.interval_minutes} minutes")
                await asyncio.sleep(self.interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                # Wait 5 minutes before retrying on error
                await asyncio.sleep(300)
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        logger.info("🛑 News Pipeline Scheduler stopped")
    
    async def _run_pipeline_cycle(self):
        """Run one cycle of the pipeline."""
        try:
            cycle_start = datetime.utcnow()
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Starting news pipeline cycle at {cycle_start}")
            logger.info(f"{'='*60}")
            
            # Run pipeline for all active alerts
            result = await self.pipeline.process_all_active_alerts()
            
            cycle_end = datetime.utcnow()
            duration = (cycle_end - cycle_start).total_seconds()
            
            if result:
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ Pipeline cycle complete!")
                logger.info(f"   Duration: {duration:.2f} seconds")
                logger.info(f"   Alerts processed: {result['successful']}/{result['total_alerts']}")
                logger.info(f"   Articles queued: {result['total_articles']}")
                logger.info(f"{'='*60}\n")
            else:
                logger.warning(f"⚠️ Pipeline cycle completed with no results")
            
            self.last_run = cycle_end
            
        except Exception as e:
            logger.error(f"❌ Error in pipeline cycle: {str(e)}")
            raise
    
    async def run_once(self):
        """Run the pipeline once (for testing or manual triggers)."""
        logger.info("🔧 Running pipeline manually (one-time)")
        await self._run_pipeline_cycle()
    
    async def cleanup(self):
        """Clean up resources."""
        await self.pipeline.close()
        logger.info("Scheduler resources cleaned up")


# Global scheduler instance
_scheduler: Optional[NewsPipelineScheduler] = None


async def start_news_pipeline_scheduler(
    gemini_api_key: str = None,
    interval_minutes: int = 60
):
    """
    Start the news pipeline scheduler.
    
    Args:
        gemini_api_key: Google Gemini API key (or from env)
        interval_minutes: How often to run (default: 60 minutes = 1 hour)
    """
    global _scheduler
    
    # Get API key from environment if not provided
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAydFe00gWhKQYdoF7oKD6QALZMwnCfkus")
    
    # Create and start scheduler
    _scheduler = NewsPipelineScheduler(
        gemini_api_key=gemini_api_key,
        interval_minutes=interval_minutes
    )
    
    await _scheduler.start()


def stop_news_pipeline_scheduler():
    """Stop the news pipeline scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()


async def run_pipeline_manually():
    """Run the pipeline once manually (for testing)."""
    global _scheduler
    
    if not _scheduler:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAydFe00gWhKQYdoF7oKD6QALZMwnCfkus")
        _scheduler = NewsPipelineScheduler(
            gemini_api_key=gemini_api_key,
            interval_minutes=60
        )
    
    await _scheduler.run_once()


# Example usage for testing
if __name__ == "__main__":
    async def test():
        # Set your API key
        api_key = "YOUR_GEMINI_API_KEY"
        
        # Create scheduler (run every 10 minutes for testing)
        scheduler = NewsPipelineScheduler(
            gemini_api_key=api_key,
            interval_minutes=10
        )
        
        # Run once
        await scheduler.run_once()
        
        # Or start continuous scheduling
        # await scheduler.start()
        
        await scheduler.cleanup()
    
    asyncio.run(test())
