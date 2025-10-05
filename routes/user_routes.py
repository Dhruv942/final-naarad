from fastapi import APIRouter, HTTPException
from db.mongo import users_collection
from models.user import UserRegister, UserResponse
import uuid
import httpx
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# WATI Configuration
WATI_ACCESS_TOKEN = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJiZGVmNjQ0OS02NDU3LTRiNDYtOTM4Mi03YjNiYmRmMmY2NGIiLCJ1bmlxdWVfbmFtZSI6ImFjdHVhbGx5dXNlZnVsZXh0ZW5zaW9uc0BnbWFpbC5jb20iLCJuYW1laWQiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiZW1haWwiOiJhY3R1YWxseXVzZWZ1bGV4dGVuc2lvbnNAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMjgvMjAyNSAxMjowNzo1OCIsInRlbmFudF9pZCI6IjQ1ODkxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.WPoEwLq2UdUs8Rl61SklQMFQ699mj1CqQ2v7iPZunuU'
WATI_BASE_URL = 'https://live-mt-server.wati.io/458913'
# Simple welcome template without variables
WATI_WELCOME_TEMPLATE = 'welcome'
WATI_WELCOME_BROADCAST = 'welcome_051020251845'

async def send_welcome_message(country_code: str, phone_number: str):
    """Send welcome message via WATI template when user logs in"""
    try:
        whatsapp_number = f"{country_code}{phone_number}"

        logger.info(f"🔔 Sending welcome message to {whatsapp_number}")

        headers = {
            'Authorization': WATI_ACCESS_TOKEN,
            'Content-Type': 'application/json-patch+json'
        }

        # Simple template without variables - static message
        payload = {
            "receivers": [
                {
                    "whatsappNumber": whatsapp_number
                }
            ],
            "template_name": WATI_WELCOME_TEMPLATE,
            "broadcast_name": WATI_WELCOME_BROADCAST
        }

        logger.info(f"📤 WATI Payload: {payload}")

        api_endpoint = f"{WATI_BASE_URL}/api/v1/sendTemplateMessages"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_endpoint,
                headers=headers,
                json=payload
            )

            logger.info(f"WATI Response Status: {response.status_code}")
            logger.info(f"WATI Response: {response.text}")

            if response.status_code == 200:
                response_data = response.json()
                wati_result = response_data.get("result", False)

                if wati_result:
                    logger.info(f"✅ Welcome message sent successfully to {whatsapp_number}")
                    return {"status": "success"}
                else:
                    logger.error(f"❌ WATI error: {response_data.get('errors', {})}")
                    return {"status": "error", "message": response_data.get('errors', {})}
            else:
                logger.error(f"Failed to send welcome message: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}

    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/login", response_model=UserResponse)
async def login_user(user: UserRegister):

    existing_user = await users_collection.find_one({
        "country_code": user.country_code,
        "phone_number": user.phone_number,
        "email": user.email,

    })

    if existing_user:
        # Send welcome message for existing user login
        await send_welcome_message(user.country_code, user.phone_number)

        return UserResponse(
            user_id=existing_user["user_id"],
            country_code=existing_user["country_code"],
            phone_number=existing_user["phone_number"],
            email=existing_user["email"],

        )

    # Else create new user
    new_user = user.dict()
    new_user["user_id"] = str(uuid.uuid4())
    await users_collection.insert_one(new_user)

    # Send welcome message for new user
    await send_welcome_message(user.country_code, user.phone_number)

    return UserResponse(
        user_id=new_user["user_id"],
        country_code=user.country_code,
        phone_number=user.phone_number,
        email=user.email,

    )
