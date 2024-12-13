# src/routers/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from loguru import logger
from src.services.security import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, users_db
from src.schemas.auth import Token

router = APIRouter(
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
)

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Informations d'identification incorrectes",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/client", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if user and user["password"] == form_data.password:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]},
            expires_delta=access_token_expires
        )
        logger.info(f"Received username: {form_data.username}")
        logger.info(f"Received password: {form_data.password}")
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Informations d'identification incorrectes",
        headers={"WWW-Authenticate": "Bearer"},
    )