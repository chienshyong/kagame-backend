from fastapi import APIRouter, Depends, HTTPException, status
import services.kagameDB as kagameDB
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from services.auth import *

router = APIRouter()


@router.post("/register")
async def register(user: User):
    if services.kagameDB.users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    services.kagameDB.users.insert_one({"username": user.username, "password": hashed_password})
    return {"msg": "User registered successfully"}


@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db_user = services.kagameDB.users.find_one({"username": form_data.username})
    if not db_user or not verify_password(form_data.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    access_token = create_access_token(data={"username": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


# Example of a protected route. returns the user's username
# Requires header 'Authorization' : 'Bearer <token>'
# TODO: Remove this route example is no longer needed.
@router.get("/username")
async def login(current_user: str = Depends(get_current_user)):
    print(current_user)
    return {"username": current_user['username']}
