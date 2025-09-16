from models.alerts import AlertCreate, AlertResponse
from fastapi import HTTPException
from bson import ObjectId
from db.mongo import alerts_collection


async def create_alert(alert_data: AlertCreate):
    sub_categories = (
        None
        if not alert_data.sub_categories or "No Preference" in alert_data.sub_categories
        else alert_data.sub_categories
    )

    new_alert = {
        "user_id": alert_data.user_id,
        "main_category": alert_data.main_category,
        "sub_categories": sub_categories,
        "is_active": True,
    }

    result = await alerts_collection.insert_one(new_alert)
    new_alert["_id"] = str(result.inserted_id)
    return AlertResponse(**new_alert)


async def get_alerts_by_user(user_id: str):
    alerts_cursor = alerts_collection.find({"user_id": user_id})
    alerts = []
    async for alert in alerts_cursor:
        alert["_id"] = str(alert["_id"])  # ✅ map _id → alert_id
        alerts.append(AlertResponse(**alert))
    return alerts


async def update_alert_by_id(user_id: str, alert_id: str, update_data: dict):
    if "sub_categories" in update_data:
        if (
            not update_data["sub_categories"]
            or "No Preference" in update_data["sub_categories"]
        ):
            update_data["sub_categories"] = None

    result = await alerts_collection.update_one(
        {"user_id": user_id, "_id": ObjectId(alert_id)}, {"$set": update_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found for this user")

    updated = await alerts_collection.find_one(
        {"user_id": user_id, "_id": ObjectId(alert_id)}
    )
    updated["_id"] = str(updated["_id"])
    return AlertResponse(**updated)


async def delete_alert_by_id(user_id: str, alert_id: str):
    result = await alerts_collection.delete_one(
        {"user_id": user_id, "_id": ObjectId(alert_id)}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found for this user")
    return {"message": "Alert deleted successfully"}

async def paused_alert_by_id(user_id:str, alert_id:str):
    result = await alerts_collection.update_one(
        {"user_id": user_id, "_id": ObjectId(alert_id)}, {"$set": {"is_active": False}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found for this user")

    updated = await alerts_collection.find_one(
        {"user_id": user_id, "_id": ObjectId(alert_id)}
    )
    updated["_id"] = str(updated["_id"])
    return AlertResponse(**updated)