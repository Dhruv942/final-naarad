from fastapi import APIRouter, HTTPException
from db.mongo import notifications_collection
from models.notification_prefernces import NotificationPreference
import uuid

router = APIRouter()

@router.post("/{user_id}")
async def set_notification_pref(user_id: str, pref: NotificationPreference):
    """Set or update a user's notification preference"""
    data = pref.dict()

    update_result = await notifications_collection.update_one(
        {"_id": user_id},
        {"$set": {"notification_preference": data}},
        upsert=True
    )

    return {"message": "Notification preference saved", "data": data}


@router.get("/{user_id}")
async def get_notification_pref(user_id: str):
    """Get a user's notification preference"""
    user = await notifications_collection.find_one({"_id": user_id}, {"notification_preference": 1, "_id": 0})
    
    if not user or "notification_preference" not in user:
        raise HTTPException(status_code=404, detail="No notification preference found")

    return {"user_id": user_id, "notification_preference": user["notification_preference"]}