from fastapi import Depends, HTTPException, status, APIRouter
import services.mongodb as mongodb
from services.mongodb import UserItem
from pydantic import BaseModel
from io import BytesIO
from services.user import get_current_user
from bson import ObjectId
from typing import Optional

router = APIRouter()

class UserProfile(BaseModel):
    gender: Optional[str] = None
    birthday: Optional[str] = None
    location: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    ethnicity: Optional[str] = None
    skin_tone: Optional[str] = None
    style: Optional[str] = None
    happiness_current_wardrobe: Optional[str] = None

@router.get("/profile/retrieve", response_model=UserProfile)
async def get_user_profile(current_user: UserItem = Depends(get_current_user)):
    query = {"_id": current_user['_id']}
    item = mongodb.users.find_one({"_id": current_user["_id"]})
    userprofile = item.get("userdefined_profile")
    if not userprofile:
        userprofile = {}
    return UserProfile(**userprofile)

@router.post("/profile/update", response_model=UserProfile)
async def update_user_profile(updated_profile: UserProfile, current_user: dict = Depends(get_current_user)):  
    users_collection = mongodb.users
    updated_profile = updated_profile.dict()
    if updated_profile["height"] == "null":
        updated_profile["height"] == ""
    if updated_profile["weight"] == "null":
        updated_profile["weight"] == ""
    updated_profile = UserProfile(**updated_profile)
    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"userdefined_profile": updated_profile.dict(exclude_none=True)}}
    )
    return updated_profile