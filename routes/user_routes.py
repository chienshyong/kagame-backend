from fastapi import APIRouter, Depends, HTTPException, Request
import services.mongodb as mongodb
from firebase_admin import auth, credentials, initialize_app
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from services.user import *
router = APIRouter()


@router.post("/register")
async def register(user: User):
    if mongodb.users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    mongodb.users.insert_one({"username": user.username, "password": hashed_password})
    return {"msg": "User registered successfully"}


@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db_user = mongodb.users.find_one({"username": form_data.username})
    if not db_user or not verify_password(form_data.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    access_token = create_access_token(data={"username": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Only initialize once
cred = credentials.Certificate("secretstuff/kagame-432309-firebase-adminsdk-fbsvc-63bdbd6563.json")
initialize_app(cred)


@router.post("/googlelogin")
async def login_with_google(request: Request):
    data = await request.json()
    id_token = data.get("id_token")

    try:
        decoded_token = auth.verify_id_token(id_token)
        email = decoded_token.get("email") or decoded_token.get("uid")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    # Optionally check if the user exists in MongoDB
    user = mongodb.users.find_one({"username": email})
    if not user:
        # Create user or return error
        user = {
            "username": email,
            "password": None,  # No password, since it's Google-authenticated
        }
        mongodb.users.insert_one(user)

    # Issue JWT compatible with existing auth logic
    token = create_access_token({"username": email})
    return {"access_token": token, "token_type": "bearer"}


@router.delete("/deleteuser")
async def delete_user(current_user: str = Depends(get_current_user)):
    user_result = mongodb.users.delete_one({"_id": current_user["_id"]})
    if user_result.deleted_count == 0:
        print(f"User {current_user['username']} with id {current_user['_id']} not deleted")
        raise HTTPException(status_code=400, detail="Deletion unsuccessful")

    wardrobe_result = mongodb.wardrobe.delete_many({"user_id": current_user['_id']})
    print(f"Successfully deleted {wardrobe_result.deleted_count} objects from wardrobe")

    return {"msg": f"User deleted and {wardrobe_result.deleted_count} objects from wardrobe deleted"}
