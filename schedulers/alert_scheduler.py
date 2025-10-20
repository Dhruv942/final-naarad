"""
Alert Scheduler - Sends scheduled WhatsApp notifications based on user alert timings
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import pytz

from controllers.alerts_controller import get_scheduled_alerts, update_alert_last_sent
from controllers.rag_news_controller_refactored import rag_system, get_alert_specific_articles
from services.rag_news.rss_fetcher import fetch_all_news
from services.rag_news.article_filter import build_contextual_query
from services.rag_news.gemini_service import final_gemini_perfect_filter
from db.mongo import db

logger = logging.getLogger(__name__)


class AlertScheduler:
    """Handles scheduled alert notifications"""

    def __init__(self):
        self.running = False

    async def start_scheduler(self):
        """Start the scheduler loop"""
        self.running = True
        logger.info("Alert Scheduler started")

        while self.running:
            try:
                await self.process_scheduled_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)

    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Alert Scheduler stopped")

    async def process_scheduled_alerts(self):
        """Process all scheduled alerts that need to be sent now"""
        try:
            # Process recurring scheduled alerts
            alerts = await get_scheduled_alerts()

            for alert in alerts:
                try:
                    if await self.should_send_alert(alert):
                        await self.send_scheduled_alert(alert)
                except Exception as e:
                    logger.error(f"Error processing alert {alert.get('alert_id')}: {e}")

            # Process one-time scheduled notifications
            await self.process_one_time_notifications()

        except Exception as e:
            logger.error(f"Error in process_scheduled_alerts: {e}")

    async def process_one_time_notifications(self):
        """Process one-time scheduled notifications that are due"""
        try:
            from controllers.alerts_controller import get_pending_scheduled_notifications, mark_notification_as_sent, mark_notification_as_failed

            # Get all pending notifications that are due
            pending_notifications = await get_pending_scheduled_notifications()

            for notification in pending_notifications:
                try:
                    await self.send_one_time_notification(notification)
                    # Delete notification after successful sending
                    await mark_notification_as_sent(notification["_id"])
                    logger.info(f"One-time notification sent and deleted: {notification['_id']}")

                except Exception as e:
                    logger.error(f"Error sending one-time notification {notification['_id']}: {e}")
                    # Mark as failed for retry logic (optional)
                    await mark_notification_as_failed(notification["_id"])

        except Exception as e:
            logger.error(f"Error in process_one_time_notifications: {e}")

    async def should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if an alert should be sent now based on its schedule"""
        try:
            schedule = alert.get("schedule", {})
            frequency = schedule.get("frequency", "realtime")

            # Realtime alerts are handled by the regular API endpoint
            if frequency == "realtime":
                return False

            # Get current time in user's timezone
            user_timezone = schedule.get("timezone", "Asia/Kolkata")
            tz = pytz.timezone(user_timezone)
            current_time = datetime.now(tz)

            # Get last sent time
            last_sent = alert.get("last_sent")
            if last_sent:
                last_sent = last_sent.replace(tzinfo=timezone.utc).astimezone(tz)

            if frequency == "hourly":
                return await self.should_send_hourly(current_time, last_sent)
            elif frequency == "daily":
                scheduled_time = schedule.get("time", "09:00")
                return await self.should_send_daily(current_time, last_sent, scheduled_time)
            elif frequency == "weekly":
                scheduled_time = schedule.get("time", "09:00")
                scheduled_days = schedule.get("days", [])
                return await self.should_send_weekly(current_time, last_sent, scheduled_time, scheduled_days)

            return False

        except Exception as e:
            logger.error(f"Error in should_send_alert: {e}")
            return False

    async def should_send_hourly(self, current_time: datetime, last_sent: datetime = None) -> bool:
        """Check if hourly alert should be sent"""
        if not last_sent:
            return True

        # Send if more than 1 hour has passed
        return current_time - last_sent >= timedelta(hours=1)

    async def should_send_daily(self, current_time: datetime, last_sent: datetime = None, scheduled_time: str = "09:00") -> bool:
        """Check if daily alert should be sent"""
        try:
            # Parse scheduled time
            hour, minute = map(int, scheduled_time.split(":"))
            scheduled_today = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Check if we're within 1 minute of the scheduled time
            time_diff = abs((current_time - scheduled_today).total_seconds())
            if time_diff > 60:  # Not within 1 minute of scheduled time
                return False

            # Check if we already sent today
            if last_sent:
                if last_sent.date() == current_time.date():
                    return False

            return True

        except Exception as e:
            logger.error(f"Error in should_send_daily: {e}")
            return False

    async def should_send_weekly(self, current_time: datetime, last_sent: datetime = None,
                                scheduled_time: str = "09:00", scheduled_days: List[str] = None) -> bool:
        """Check if weekly alert should be sent"""
        try:
            if not scheduled_days:
                return False

            # Check if today is a scheduled day
            current_day = current_time.strftime("%A").lower()
            if current_day not in [day.lower() for day in scheduled_days]:
                return False

            # Use daily logic for the time check
            return await self.should_send_daily(current_time, last_sent, scheduled_time)

        except Exception as e:
            logger.error(f"Error in should_send_weekly: {e}")
            return False

    async def send_scheduled_alert(self, alert: Dict[str, Any]):
        """Send a scheduled alert notification"""
        try:
            alert_id = alert.get("alert_id")
            user_id = alert.get("user_id")

            logger.info(f"Sending scheduled alert {alert_id} for user {user_id}")

            # Ensure we have fresh news data
            await self.refresh_news_if_needed()

            # Build alert query
            category = alert.get("main_category", "").lower()
            keywords = alert.get("sub_categories", [])
            followup_questions = alert.get("followup_questions", [])
            custom_question = alert.get("custom_question", "")

            alert_query = await build_contextual_query(alert, keywords, followup_questions, custom_question, category)

            # Get category-specific articles
            relevant_articles = await get_alert_specific_articles(alert, alert_query, category)

            if not relevant_articles:
                logger.info(f"No relevant articles found for scheduled alert {alert_id}")
                return

            # Apply intelligent filtering (get top 3 articles)
            contextually_relevant = await final_gemini_perfect_filter(relevant_articles, [alert], 3)

            if not contextually_relevant:
                logger.info(f"No articles passed intelligent filter for scheduled alert {alert_id}")
                return

            # Send each article as separate WhatsApp notification
            for article in contextually_relevant:
                # Create alert result structure
                alert_result = {
                    "alert_id": alert_id,
                    "alert_category": category,
                    "alert_keywords": keywords,
                    "alert_query": alert_query,
                    "total_articles": 1,
                    "articles": [article]
                }

                # Send WhatsApp notification (disabled for testing)
                wati_response = {"status": "skipped", "message": "WATI disabled in testing"}
                logger.info(f"Scheduled alert notification sent: {wati_response.get('status', 'unknown')}")

                # Add small delay between notifications
                await asyncio.sleep(1)

            # Update last sent timestamp
            await update_alert_last_sent(alert_id)
            logger.info(f"Updated last_sent for alert {alert_id}")

        except Exception as e:
            logger.error(f"Error sending scheduled alert: {e}")

    async def send_one_time_notification(self, notification: Dict[str, Any]):
        """Send a one-time scheduled notification"""
        try:
            user_id = notification.get("user_id")
            alert_id = notification.get("alert_id")
            articles = notification.get("articles", [])

            logger.info(f"Sending one-time notification for user {user_id}, alert {alert_id}")

            if not articles:
                logger.info(f"No articles to send for notification {notification['_id']}")
                return

            # Send each article as separate WhatsApp notification
            for article in articles:
                # Create alert result structure (similar to regular alerts)
                alert_result = {
                    "alert_id": alert_id,
                    "alert_category": article.get("category", ""),
                    "alert_keywords": [],  # Will be populated from stored data if needed
                    "alert_query": "",
                    "total_articles": 1,
                    "articles": [article]
                }

                # Send WhatsApp notification (disabled for testing)
                wati_response = {"status": "skipped", "message": "WATI disabled in testing"}
                logger.info(f"One-time notification sent: {wati_response.get('status', 'unknown')}")

                # Add small delay between notifications
                await asyncio.sleep(1)

            logger.info(f"All articles sent for one-time notification {notification['_id']}")

        except Exception as e:
            logger.error(f"Error sending one-time notification: {e}")
            raise

    async def refresh_news_if_needed(self):
        """Refresh news data if needed"""
        try:
            doc_count = await db.get_collection("document_vectors").count_documents({})
            if doc_count == 0:
                logger.info("No documents found, fetching fresh news...")
                articles = await fetch_all_news()
                if articles:
                    await rag_system.process_news_articles(articles)
                    logger.info(f"Fetched and processed {len(articles)} articles")
        except Exception as e:
            logger.error(f"Error in news refresh: {e}")


# Global scheduler instance
alert_scheduler = AlertScheduler()


async def start_alert_scheduler():
    """Start the alert scheduler"""
    await alert_scheduler.start_scheduler()


def stop_alert_scheduler():
    """Stop the alert scheduler"""
    alert_scheduler.stop_scheduler()