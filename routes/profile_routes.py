from fastapi import Depends, HTTPException, status, APIRouter
import services.mongodb as mongodb
from services.mongodb import UserItem
from pydantic import BaseModel
from typing import Optional, OrderedDict, Dict, Union, List, Tuple
from services.user import get_current_user
from datetime import datetime, date
import json

router = APIRouter()

FeedbackItem = Tuple[str, str, str, Union[str, List[str]]]

class UserProfile(BaseModel):
    # All fields as strings or None
    age: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[str] = None
    location: Optional[str] = None
    style: Optional[str] = None
    clothing_likes: Optional[OrderedDict[str, bool]] = {}
    clothing_dislikes: Optional[OrderedDict[str, Union[List[FeedbackItem], bool]]] = {}

class Preferences(BaseModel):
    clothing_likes: Optional[OrderedDict[str, bool]] = None
    clothing_dislikes: Optional[OrderedDict[str, Union[bool, List[List[str]]]]] = None  

class ClothingPreferenceUpdate(BaseModel):
    Dict[str, Union[bool, List]]

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
    profile_data = updated_profile.dict(exclude_unset=True)

    # If "null" is sent as a string for height/weight, convert it to empty
    if profile_data.get("height") == "null":
        profile_data["height"] = ""
    if profile_data.get("weight") == "null":
        profile_data["weight"] = ""

    existing_profile = users_collection.find_one({"_id": current_user["_id"]}, {"userdefined_profile": 1})

    # Merge the existing profile with the updated fields
    if existing_profile and "userdefined_profile" in existing_profile:
        merged_profile = {**existing_profile["userdefined_profile"], **profile_data}
    else:
        merged_profile = profile_data

    if "clothing_dislikes" in merged_profile and "feedback" in merged_profile["clothing_dislikes"]:
        for i, feedback_item in enumerate(merged_profile["clothing_dislikes"]["feedback"]):
            merged_profile["clothing_dislikes"]["feedback"][i] = [
                x if x is not None else "" for x in feedback_item
            ]

    # Re-validate after adjusting values
    validated_profile = UserProfile(**merged_profile)

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
        {"$set": {"userdefined_profile": validated_profile.dict()}}
    )

    return validated_profile

@router.get("/profile/getclothingprefs", response_model=Preferences)
async def get_clothing_prefs(current_user: UserItem = Depends(get_current_user)):
    try:    
        user = mongodb.users.find_one({"_id": current_user["_id"]})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        print(f"Error retrieving user: {e}")
    
    userprofile = user.get("userdefined_profile", {})
    
    clothing_likes = userprofile.get("clothing_likes", {})
    clothing_dislikes = userprofile.get("clothing_dislikes", {})

    return {"clothing_likes": clothing_likes, "clothing_dislikes": clothing_dislikes}

@router.post("/profile/getprefsurls")
async def get_prefs_image_URLS(prefs: dict):
    url_dict = {}
    prefs_list = list(prefs.keys())
    try:
        for name in prefs_list:
            if name == "feedback":
                continue
            item = mongodb.catalogue.find_one({"name": name})
            url_dict[name] = item["image_url"]
        return url_dict
    except Exception as e:
        print(f"Error getting prefs URLs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prefs image URLs")


@router.post("/profile/updateclothinglikes", response_model=ClothingPreferenceUpdate)
async def update_clothing_likes(updated_clothing_like: dict, 
                              current_user: dict = Depends(get_current_user)):
    try:
        user = mongodb.users.find_one({"_id": current_user["_id"]})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        userprofile = user.get("userdefined_profile", {})
        clothing_likes = userprofile.get("clothing_likes", {})
        clothing_dislikes = userprofile.get("clothing_dislikes", {})

        for item, is_liked in updated_clothing_like.items():
            if is_liked:  
                clothing_likes[item] = True
                clothing_dislikes.pop(item, None)
            else:
                clothing_likes.pop(item, None)

        mongodb.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": {
                "userdefined_profile.clothing_likes": clothing_likes,
                "userdefined_profile.clothing_dislikes": clothing_dislikes
            }}
        )

        return {"message": "Clothing preferences updated successfully"}

    except Exception as e:
        print(f"Error updating clothing preferences (likes function): {e}")
        raise HTTPException(status_code=500, detail="Failed to update clothing preferences")
    
@router.post("/profile/updateclothingdislikes", response_model=ClothingPreferenceUpdate)
async def update_clothing_dislikes(updated_clothing_dislike: dict, 
                              current_user: dict = Depends(get_current_user)):
    try:
        user = mongodb.users.find_one({"_id": current_user["_id"]})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        userprofile = user.get("userdefined_profile", {})
        clothing_likes = userprofile.get("clothing_likes", {})
        clothing_dislikes = userprofile.get("clothing_dislikes", {})

        if 'feedback' in updated_clothing_dislike:
            if 'feedback' in clothing_dislikes:
                if len(clothing_dislikes["feedback"]) >= 100:
                    del clothing_dislikes["feedback"][:95]
                clothing_dislikes["feedback"].append(updated_clothing_dislike['feedback'])
            else:
                clothing_dislikes["feedback"] = [updated_clothing_dislike['feedback']]
            updated_clothing_dislike.pop("feedback", None)

        for item, is_disliked in updated_clothing_dislike.items():
            if is_disliked:  
                clothing_dislikes[item] = True
                clothing_likes.pop(item, None)
            else:
                clothing_dislikes.pop(item, None)
                
        mongodb.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": {
                "userdefined_profile.clothing_likes": clothing_likes,
                "userdefined_profile.clothing_dislikes": clothing_dislikes
            }}
        )

        return {"message": "Clothing preferences updated successfully"}

    except Exception as e:
        print(f"Error updating clothing preferences (dislikes function): {e}")
        raise HTTPException(status_code=500, detail="Failed to update clothing preferences")
    
@router.post("/profile/stylequizresult")
async def update_style_quiz_result(updated_style_quiz_result: dict, 
                              current_user: dict = Depends(get_current_user)):
    try:
        user = mongodb.users.find_one({"_id": current_user["_id"]})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        mongodb.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": {
                "userdefined_profile.style": updated_style_quiz_result['style_result']
            }}
        )

        return {"message": "Style quiz result updated successfully"}

    except Exception as e:
        print(f"Error updating style quiz result: {e}")
        raise HTTPException(status_code=500, detail="Failed to update style quiz result")