from fastapi import Depends, HTTPException, status
import services.mongodb as mongodb
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from fastapi.security import OAuth2PasswordBearer
from services import JWT_SECRET_KEY
from typing_extensions import Annotated

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = JWT_SECRET_KEY
ALGORITHM = "HS256"


class User(BaseModel):
    username: str
    password: str


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, ALGORITHM)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, ALGORITHM)
        username: str = payload.get("username")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = mongodb.users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user
