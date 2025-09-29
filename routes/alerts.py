from fastapi import APIRouter
from typing import List
from models.alerts import AlertCreate, AlertResponse, ScheduleSettings
from controllers.alerts_controller import (
    create_alert,
    get_alerts_by_user,
    update_alert_by_id,
    delete_alert_by_id,
    paused_alert_by_id,
    update_alert_schedule,
    schedule_one_time_notification,
)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.post("/", response_model=AlertResponse)
async def post_alert(alert: AlertCreate):
    """Create a new alert"""
    return await create_alert(alert)


@router.get("/{user_id}", response_model=List[AlertResponse])
async def get_alerts(user_id: str):
    """Get all alerts for a user"""
    return await get_alerts_by_user(user_id)


@router.put("/{user_id}/{alert_id}", response_model=AlertResponse)
async def put_alert_by_id(user_id: str, alert_id: str, update_data: dict):
    """
    Update specific alert by user_id + alert_id.
    Example body:
    {
      "sub_categories": ["T20", "Notify me if RCB wins"],
      "is_active": false
    }
    """
    return await update_alert_by_id(user_id, alert_id, update_data)


@router.delete("/{user_id}/{alert_id}")
async def delete_alert(user_id: str, alert_id: str):
    """Delete an alert"""
    return await delete_alert_by_id(user_id, alert_id)


@router.put("/{user_id}/{alert_id}/pause", response_model=AlertResponse)
async def pause_alert(user_id: str, alert_id: str):
    """Pause an alert (set is_active = False)"""
    return await paused_alert_by_id(user_id, alert_id)


@router.put("/{user_id}/{alert_id}/schedule", response_model=AlertResponse)
async def update_alert_timing(user_id: str, alert_id: str, schedule_settings: ScheduleSettings):
    """
    Update alert scheduling preferences.

    Example request body:
    {
      "frequency": "daily",
      "time": "08:30",
      "timezone": "Asia/Kolkata"
    }

    Or for weekly:
    {
      "frequency": "weekly",
      "time": "09:00",
      "days": ["monday", "wednesday", "friday"],
      "timezone": "Asia/Kolkata"
    }
    """
    return await update_alert_schedule(user_id, alert_id, schedule_settings.dict())


@router.post("/{user_id}/{alert_id}/schedule-time")
async def set_one_time_schedule(user_id: str, alert_id: str, schedule_data: dict):
    """
    Set one-time notification schedule for alert.

    Example request body:
    {
        "scheduled_datetime": "2025-09-29T15:30:00",  // ISO format
        "timezone": "Asia/Kolkata"
    }

    This will:
    1. Fetch news for this alert immediately
    2. Store the articles with scheduled time
    3. Send WhatsApp at specified time
    4. Delete the scheduled notification after sending
    """
    from datetime import datetime
    import pytz
    from controllers.rag_news_controller import get_user_news

    try:
        # Parse scheduled datetime
        scheduled_time_str = schedule_data.get("scheduled_datetime")
        timezone_str = schedule_data.get("timezone", "Asia/Kolkata")

        # Convert to datetime with timezone
        tz = pytz.timezone(timezone_str)
        scheduled_datetime = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))

        # Convert to UTC for storage
        if scheduled_datetime.tzinfo is None:
            # If no timezone info, assume it's in user's timezone
            scheduled_datetime = tz.localize(scheduled_datetime)

        scheduled_datetime_utc = scheduled_datetime.astimezone(pytz.UTC)

        # Fetch current news for this alert
        news_response = await get_user_news(user_id)

        # Find articles for this specific alert
        alert_articles = []
        if news_response.get("status") == "success":
            for alert_result in news_response.get("alert_results", []):
                if alert_result.get("alert_id") == alert_id:
                    alert_articles = alert_result.get("articles", [])
                    break

        if not alert_articles:
            return {
                "status": "error",
                "message": "No articles found for this alert"
            }

        # Store scheduled notification
        scheduled_notification = await schedule_one_time_notification(
            user_id=user_id,
            alert_id=alert_id,
            scheduled_datetime=scheduled_datetime_utc,
            articles_data=alert_articles
        )

        return {
            "status": "success",
            "message": f"Notification scheduled for {scheduled_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            "scheduled_time": scheduled_datetime_utc.isoformat(),
            "articles_count": len(alert_articles),
            "notification_id": str(scheduled_notification.get("_id"))
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/trigger-scheduled")
async def trigger_scheduled_alerts():
    """
    Manual trigger for scheduled alerts (for testing purposes).
    This will process all scheduled alerts that are due to be sent.
    """
    from schedulers.alert_scheduler import alert_scheduler
    await alert_scheduler.process_scheduled_alerts()
    return {"message": "Scheduled alerts processing triggered successfully"}
