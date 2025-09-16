from fastapi import APIRouter, HTTPException
from db.mongo import users_collection
from models.user import UserRegister, UserResponse
import uuid

router = APIRouter()

@router.post("/login", response_model=UserResponse)
async def login_user(user: UserRegister):
    
    existing_user = await users_collection.find_one({
        "country_code": user.country_code,
        "phone_number": user.phone_number,
        "email": user.email,

    })

    if existing_user:
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

    return UserResponse(
        user_id=new_user["user_id"],  
        country_code=user.country_code,
        phone_number=user.phone_number,
        email=user.email,
   
    )
