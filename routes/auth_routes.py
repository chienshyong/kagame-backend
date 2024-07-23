from fastapi import APIRouter, Depends, HTTPException, status
import kagameDB
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from fastapi.security import OAuth2PasswordBearer
from routes.auth import *

router = APIRouter()

@router.post("/register")
async def register(user: User):
    if kagameDB.users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    kagameDB.users.insert_one({"username": user.username, "password": hashed_password})
    return {"msg": "User registered successfully"}

@router.post("/login")
async def login(user: User):
    db_user = kagameDB.users.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token}

#Example of a protected route
#Requires header 'Authorization' : 'Bearer <token>'
@router.get("/username")
async def login(current_user: dict = Depends(get_current_user)):
    print(current_user)
    return {"username" : current_user['username']}