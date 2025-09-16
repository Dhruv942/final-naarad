from pydantic import BaseModel, Field
from typing import List, Optional
class UserRegister(BaseModel):
    country_code: str = Field(..., example="+91")
    phone_number: str = Field(..., example="9876543210")
    email:str = Field(..., example ="user@gmail.com")
    

class UserResponse(BaseModel):
    user_id: str
    country_code: str
    phone_number: str
    email: str


class Userprefernces(BaseModel):
    user_id:str