import logging

logger = logging.getLogger(__name__)

# यह सिर्फ demo है (test के लिए print/log कर रहा हूँ)
# Production में इसे तू FCM / OneSignal / WhatsApp API / Email वगैरह से replace कर सकता है
async def send_notification(user_id: str, alert_id: str, news: dict):
    """
    Automatically called when a personalized news is found
    """
    logger.info(f"🔔 Sending notification to user={user_id}, alert={alert_id}")
    logger.info(f"News: {news['title']} -> {news['link']}")

    # Future: यहां पर actual API call (FCM, Email, WhatsApp) कर सकते हैं
    return {"status": "sent", "user_id": user_id, "alert_id": alert_id}
