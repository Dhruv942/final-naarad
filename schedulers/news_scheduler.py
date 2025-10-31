"""
Alternating News Scheduler

Runs the personalized news pipeline in alternating windows:
 - Active window: run for N minutes (default 30) → process all active alerts once
 - Sleep window: sleep for M minutes (default 30)

Notes:
 - Sends only when relevant articles are found (pipeline handles this)
 - No compulsory messages; if nothing relevant, nothing is sent
"""

import asyncio
import logging
from datetime import datetime, timedelta
import os

from core.personalized_news_pipeline import PersonalizedNewsPipeline
from db.mongo import db

logger = logging.getLogger(__name__)


async def _run_once(pipeline: PersonalizedNewsPipeline) -> None:
    """Run one full batch over all active alerts."""
    try:
        start_ts = datetime.utcnow()
        logger.info("[CRON] Batch run start")
        await pipeline.process_all_active_alerts()
        end_ts = datetime.utcnow()
        logger.info(f"[CRON] Batch run complete in {(end_ts - start_ts).total_seconds():.1f}s")
    except Exception as e:
        logger.error(f"[CRON] Batch run error: {e}")


async def start_alternating_scheduler(
    run_minutes: int = 30,
    sleep_minutes: int = 30,
    model: str = "gemini-2.5-flash",
) -> None:
    """
    Start the alternating scheduler loop.

    Active window: run once (process all active alerts) and remain idle until window ends.
    Sleep window: pause for configured minutes.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    pipeline = PersonalizedNewsPipeline(db=db, gemini_api_key=api_key, model=model)

    try:
        while True:
            try:
                # Active window
                logger.info(f"[CRON] Active window start ({run_minutes} minutes)")
                window_end = datetime.utcnow() + timedelta(minutes=run_minutes)

                # Run once at the start of the active window
                await _run_once(pipeline)

                # If there's time left in the window, idle (no forced sends)
                remaining = (window_end - datetime.utcnow()).total_seconds()
                if remaining > 0:
                    await asyncio.sleep(remaining)

                logger.info("[CRON] Active window end")

            except asyncio.CancelledError:
                logger.info("[CRON] Alternating scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"[CRON] Alternating scheduler error: {e}")

            # Sleep window
            logger.info(f"[CRON] Sleep window start ({sleep_minutes} minutes)")
            try:
                await asyncio.sleep(max(0, sleep_minutes * 60))
            except asyncio.CancelledError:
                logger.info("[CRON] Alternating scheduler cancelled during sleep")
                break
            logger.info("[CRON] Sleep window end")
    finally:
        try:
            await pipeline.close()
        except Exception:
            pass
        logger.info("[CRON] Scheduler resources cleaned up")


