from fastapi import HTTPException
from bson import ObjectId
from db.mongo import alerts_collection
from models.alerts import AlertCreate, AlertResponse
import uuid
from datetime import datetime


async def create_alert(alert_data: AlertCreate):
    new_alert = {
        "alert_id": str(uuid.uuid4()),  # ✅ custom alert_id
        "user_id": alert_data.user_id,
        "main_category": alert_data.main_category,
        "sub_categories": alert_data.sub_categories or None,
        "followup_questions": alert_data.followup_questions or None,
        "custom_question": alert_data.custom_question,
        "is_active": True,
        # Default scheduling settings
        "schedule": {
            "frequency": "realtime",  # realtime, hourly, daily, weekly
            "time": "09:00",  # for daily/weekly alerts
            "days": [],  # for weekly alerts: ["monday", "friday"]
            "timezone": "Asia/Kolkata"
        },
        "last_sent": None,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }

    await alerts_collection.insert_one(new_alert)
    return AlertResponse(**new_alert)


async def get_alerts_by_user(user_id: str):
    alerts_cursor = alerts_collection.find({"user_id": user_id})
    alerts = []
    async for alert in alerts_cursor:
        alerts.append(AlertResponse(**alert))
    return alerts


async def update_alert_by_id(user_id: str, alert_id: str, update_data: dict):
    if "sub_categories" in update_data:
        if (
            not update_data["sub_categories"]
            or "No Preference" in update_data["sub_categories"]
        ):
            update_data["sub_categories"] = None

    if "followup_questions" in update_data:
        if not update_data["followup_questions"]:
            update_data["followup_questions"] = None

    result = await alerts_collection.update_one(
        {"user_id": user_id, "alert_id": alert_id}, {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found for this user")

    updated = await alerts_collection.find_one(
        {"user_id": user_id, "alert_id": alert_id}
    )
    return AlertResponse(**updated)


async def delete_alert_by_id(user_id: str, alert_id: str):
    result = await alerts_collection.delete_one(
        {"user_id": user_id, "alert_id": alert_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found for this user")
    return {"message": "Alert deleted successfully"}


async def paused_alert_by_id(user_id: str, alert_id: str):
    result = await alerts_collection.update_one(
        {"user_id": user_id, "alert_id": alert_id}, {"$set": {"is_active": False}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found for this user")

    updated = await alerts_collection.find_one(
        {"user_id": user_id, "alert_id": alert_id}
    )
    return AlertResponse(**updated)


async def update_alert_schedule(user_id: str, alert_id: str, schedule_data: dict):
    """Update alert scheduling preferences"""

    # Validate schedule data
    valid_frequencies = ["realtime", "hourly", "daily", "weekly"]
    if schedule_data.get("frequency") not in valid_frequencies:
        raise HTTPException(status_code=400, detail="Invalid frequency")

    update_data = {
        "schedule": schedule_data,
        "updated_at": datetime.now()
    }

    result = await alerts_collection.update_one(
        {"user_id": user_id, "alert_id": alert_id},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found for this user")

    updated = await alerts_collection.find_one(
        {"user_id": user_id, "alert_id": alert_id}
    )
    return AlertResponse(**updated)


async def get_scheduled_alerts():
    """Get all active alerts for cron job processing"""
    alerts_cursor = alerts_collection.find({"is_active": True})
    alerts = []
    async for alert in alerts_cursor:
        # Convert MongoDB document to dict and ensure proper field mapping
        alert_dict = dict(alert)

        # Ensure alert_id exists (handle legacy alerts that might not have it)
        if "alert_id" not in alert_dict and "_id" in alert_dict:
            alert_dict["alert_id"] = str(alert_dict["_id"])

        # Remove MongoDB _id to avoid confusion
        if "_id" in alert_dict:
            del alert_dict["_id"]

        alerts.append(alert_dict)
    return alerts


async def update_alert_last_sent(alert_id: str):
    """Update last_sent timestamp after sending WhatsApp"""
    await alerts_collection.update_one(
        {"alert_id": alert_id},
        {"$set": {"last_sent": datetime.now()}}
    )
