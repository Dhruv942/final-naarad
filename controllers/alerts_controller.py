from fastapi import HTTPException
from bson import ObjectId
from db.mongo import alerts_collection
from models.alerts import AlertCreate, AlertResponse
import uuid


async def create_alert(alert_data: AlertCreate):
    new_alert = {
        "alert_id": str(uuid.uuid4()),  # ✅ custom alert_id
        "user_id": alert_data.user_id,
        "main_category": alert_data.main_category,
        "sub_categories": alert_data.sub_categories or None,
        "followup_questions": alert_data.followup_questions or None,
        "custom_question": alert_data.custom_question,
        "is_active": True,
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
