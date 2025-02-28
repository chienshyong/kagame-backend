from fastapi import Depends, HTTPException, status, APIRouter
import services.mongodb as mongodb
from services.mongodb import UserItem
from pydantic import BaseModel
from typing import Optional
from services.user import get_current_user
from datetime import datetime, date

router = APIRouter()

class UserProfile(BaseModel):
    # All fields as strings or None
    age: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[str] = None
    location: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    ethnicity: Optional[str] = None
    skin_tone: Optional[str] = None
    style: Optional[str] = None
    happiness_current_wardrobe: Optional[str] = None
    clothing_preferences: Optional[str] = None 
    clothing_dislikes: Optional[str] = None   

@router.get("/profile/retrieve", response_model=UserProfile)
async def get_user_profile(current_user: UserItem = Depends(get_current_user)):
    item = mongodb.users.find_one({"_id": current_user["_id"]})
    userprofile = item.get("userdefined_profile", {})
    
    # Attempt to calculate 'age' from 'birthday'
    birthday_str = userprofile.get("birthday")  # e.g. "dd/mm/yyyy"
    calculated_age = ""
    if birthday_str:
        try:
            bday = datetime.strptime(birthday_str, "%d/%m/%Y").date()
            today = date.today()
            age_int = today.year - bday.year - ((today.month, today.day) < (bday.month, bday.day))
            calculated_age = str(age_int)
        except ValueError:
            # If parsing fails, fallback to empty string
            pass

    userprofile["age"] = calculated_age
    return UserProfile(**userprofile)

@router.post("/profile/update", response_model=UserProfile)
async def update_user_profile(updated_profile: UserProfile, 
                              current_user: dict = Depends(get_current_user)):
    users_collection = mongodb.users
    profile_data = updated_profile.dict()

    # If "null" is sent as a string for height/weight, convert it to empty
    if profile_data.get("height") == "null":
        profile_data["height"] = ""
    if profile_data.get("weight") == "null":
        profile_data["weight"] = ""

    # Re-validate after adjusting values
    validated_profile = UserProfile(**profile_data)

    # Calculate age based on birthday
    if validated_profile.birthday:
        try:
            bday = datetime.strptime(validated_profile.birthday, "%d/%m/%Y").date()
            today = date.today()
            age_int = today.year - bday.year - ((today.month, today.day) < (bday.month, bday.day))
            validated_profile.age = str(age_int)
        except ValueError:
            validated_profile.age = ""

    # Save to the database
    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"userdefined_profile": validated_profile.dict(exclude_none=True)}}
    )

    return validated_profile
