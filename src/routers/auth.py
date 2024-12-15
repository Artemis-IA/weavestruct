from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from src.services.security import authenticate_user, create_access_token, get_user_by_username, get_password_hash, verify_password
from src.config import settings
from src.schemas.auth import Token, UserCreate, UserRead
from src.utils.database import DatabaseUtils
from src.models.auth import User

router = APIRouter(
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
)

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(DatabaseUtils.get_db)):
    try:
        user = authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except SQLAlchemyError as e:
        logger.error(f"Database error during authentication: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/client", response_model=Token)
def login_client(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(DatabaseUtils.get_db),
):
    try:
        user = get_user_by_username(db, form_data.username)
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except SQLAlchemyError as e:
        logger.error(f"Database error during client login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    except Exception as e:
        logger.error(f"Unexpected error during client login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/register", response_model=UserRead)
def register_user(
    user: UserCreate, 
    db: Session = Depends(DatabaseUtils.get_db)
):
    try:
        existing_user = db.query(User).filter(User.username == user.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )
        hashed_password = get_password_hash(user.password)
        new_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            is_active=True
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except SQLAlchemyError as e:
        logger.error(f"Database error during user registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    except Exception as e:
        logger.error(f"Unexpected error during user registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/reset-password")
def reset_password(
    username: str, 
    new_password: str, 
    db: Session = Depends(DatabaseUtils.get_db)
):
    try:
        user = db.query(User).filter(User.username == username).first()
        hashed_password = get_password_hash(new_password)
        if not user:
            new_user = User(
                username=username, 
                email=f"{username}@example.com", 
                hashed_password=hashed_password, 
                is_active=True
            )
            db.add(new_user)
            db.commit()
            return {"message": f"User {username} created successfully with a new password"}
        user.hashed_password = hashed_password
        db.commit()
        return {"message": f"Password for {username} updated successfully"}
    except SQLAlchemyError as e:
        logger.error(f"Database error during password reset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    except Exception as e:
        logger.error(f"Unexpected error during password reset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
