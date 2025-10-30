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
        model: str = "gemini-2.5-flash"
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
        logger.info("[CRON] News pipeline scheduler started")
        
        while self.running:
            try:
                await self._run_pipeline_cycle()
                
                # Wait for next cycle
                logger.info(f"[CRON] Next run in {self.interval_minutes} minutes")
                await asyncio.sleep(self.interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info("[CRON] News pipeline scheduler cancelled")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                # Wait 5 minutes before retrying on error
                await asyncio.sleep(300)

    async def start_burst(self, burst_minutes: int = 8):
        """Run the scheduler for a fixed burst window, then stop.

        Keeps the task alive for `burst_minutes` while running at least one cycle.
        """
        self.running = True
        window_end = datetime.utcnow() + timedelta(minutes=burst_minutes)
        logger.info(f"[CRON] Burst start window {burst_minutes}m")
        try:
            # Always run at least one cycle
            await self._run_pipeline_cycle()
            # Stay alive (lightweight wait) until window end
            while datetime.utcnow() < window_end and self.running:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("[CRON] Burst cancelled")
        except Exception as e:
            logger.error(f"[CRON] Burst error: {e}")
        finally:
            self.running = False
            logger.info("[CRON] Burst over")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        logger.info("🛑 News Pipeline Scheduler stopped")
    
    async def _run_pipeline_cycle(self):
        """Run one cycle of the pipeline."""
        try:
            cycle_start = datetime.utcnow()
            logger.info(f"[CRON] Cycle start {cycle_start.isoformat()}")
            
            # Run pipeline for all active alerts
            result = await self.pipeline.process_all_active_alerts()
            
            cycle_end = datetime.utcnow()
            duration = (cycle_end - cycle_start).total_seconds()
            
            logger.info(f"[CRON] Cycle over (duration {duration:.2f}s)")
            
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


async def start_news_pipeline_bursts(
    gemini_api_key: str = None,
    interval_minutes: int = 60,
    burst_minutes: int = 8,
):
    """Continuously run bursts: run for burst_minutes, then wait until the next interval.

    Example: burst 8 minutes, then sleep 52 minutes => repeats hourly.
    """
    global _scheduler
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAydFe00gWhKQYdoF7oKD6QALZMwnCfkus")

    while True:
        try:
            _scheduler = NewsPipelineScheduler(
                gemini_api_key=gemini_api_key,
                interval_minutes=interval_minutes,
            )
            await _scheduler.start_burst(burst_minutes=burst_minutes)
        except Exception as e:
            logger.error(f"[CRON] Burst runner error: {e}")
        # Sleep the remainder of the interval
        sleep_secs = max(0, (interval_minutes - burst_minutes) * 60)
        logger.info(f"[CRON] Next burst in {sleep_secs // 60} minutes")
        await asyncio.sleep(sleep_secs)


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


async def start_news_pipeline_alternating(
    gemini_api_key: str = None,
    run_minutes: int = 1,
    sleep_minutes: int = 1
):
    """Run scheduler alternating: ON for run_minutes, OFF for sleep_minutes.
    
    Example: run 1 minute, sleep 1 minute, repeat.
    """
    global _scheduler
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAydFe00gWhKQYdoF7oKD6QALZMwnCfkus")

    while True:
        try:
            logger.info(f"[CRON] Starting pipeline run for {run_minutes} minutes...")
            _scheduler = NewsPipelineScheduler(
                gemini_api_key=gemini_api_key,
                interval_minutes=run_minutes
            )
            await _scheduler.start_burst(burst_minutes=run_minutes)
        except asyncio.CancelledError:
            logger.info("[CRON] Alternating scheduler cancelled")
            break
        except Exception as e:
            logger.error(f"[CRON] Alternating scheduler error: {e}")
        
        # Sleep for sleep_minutes
        logger.info(f"[CRON] Sleeping for {sleep_minutes} minutes...")
        await asyncio.sleep(sleep_minutes * 60)


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
