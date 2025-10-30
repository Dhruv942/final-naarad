import logging
import os
import httpx
from typing import Dict, Any
from db.mongo import users_collection
from services.rag_news.config import (
    WATI_ACCESS_TOKEN as CFG_WATI_ACCESS_TOKEN,
    WATI_BASE_URL as CFG_WATI_BASE_URL,
    WATI_TEMPLATE_NAME as CFG_WATI_TEMPLATE_NAME,
    WATI_BROADCAST_NAME as CFG_WATI_BROADCAST_NAME,
)

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


# WATI WhatsApp integration for sending article updates (prefer config.py values)
WATI_ACCESS_TOKEN = CFG_WATI_ACCESS_TOKEN or os.getenv("WATI_ACCESS_TOKEN")
WATI_BASE_URL = CFG_WATI_BASE_URL or os.getenv("WATI_BASE_URL")  # e.g., https://live-mt-server.wati.io/458913


async def send_wati_notification(user_id: str, alert_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a WhatsApp message via WATI with the article details for a user's alert.

    alert_payload example:
    {
      "alert_id": str,
      "alert_category": str,
      "alert_keywords": list,
      "total_articles": 1,
      "articles": [ { title, url, description, source, image } ]
    }
    """
    try:
        if not WATI_ACCESS_TOKEN or not WATI_BASE_URL:
            logger.info("WATI config missing; skipping WhatsApp send")
            return {"status": "skipped", "reason": "missing_config"}

        # Find user's phone details
        user = await users_collection.find_one({"user_id": user_id})
        if not user:
            return {"status": "error", "message": "user_not_found"}

        country_code = (user.get("country_code") or "+91").strip()
        phone_number = (user.get("phone_number") or "").strip()
        if not phone_number:
            return {"status": "error", "message": "phone_missing"}

        # Normalize to digits only for WATI (WhatsApp number usually E.164 without '+')
        import re
        cc_digits = re.sub(r"\D", "", country_code)
        pn_digits = re.sub(r"\D", "", phone_number)
        whatsapp_number = f"{cc_digits}{pn_digits}"

        article = (alert_payload.get("articles") or [{}])[0]
        title = (article.get("title") or "News Update").strip()
        description = (article.get("description") or "").strip()
        image_url = (article.get("image") or "").strip()
        source = (article.get("source") or "").strip()

        headers = {
            "Authorization": WATI_ACCESS_TOKEN,
            "Content-Type": "application/json-patch+json",
        }

        # Use template with variables: {{1}} image, {{2}} title, {{3}} description
        template_name = CFG_WATI_TEMPLATE_NAME or os.getenv("WATI_TEMPLATE_NAME", "personalized_update")
        broadcast_name = CFG_WATI_BROADCAST_NAME or os.getenv("WATI_BROADCAST_NAME")

        # WATI API uses /api/v1/sendTemplateMessages (plural) with receivers array
        # Template format: Image, Title, Description (3 variables)
        # Use customParams inside receivers array (not parameters)
        payload = {
            "receivers": [
                {
                    "whatsappNumber": whatsapp_number,
                    "customParams": [
                        {"name": "1", "value": image_url or ""},
                        {"name": "2", "value": title or ""},
                        {"name": "3", "value": description[:500] if description else ""},
                     
                    ]
                }
            ],
            "template_name": template_name,
            "broadcast_name": broadcast_name
        }
        
        endpoint = f"{WATI_BASE_URL}/api/v1/sendTemplateMessages"

        logger.info(f"📤 WATI Request: {endpoint}")
        logger.info(f"📤 WATI Payload: {payload}")

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            
            logger.info(f"📥 WATI Response: {resp.status_code} - {resp.text}")
            
            return {
                "status": "success" if resp.status_code == 200 else "error",
                "code": resp.status_code,
                "response": resp.text
            }

    except Exception as e:
        logger.error(f"WATI integration error: {e}")
        return {"status": "error", "message": str(e)}
