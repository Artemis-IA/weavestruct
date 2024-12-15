from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from uuid import uuid4

class UserBase(BaseModel):
    id: uuid4 = Field(default_factory=uuid4)
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None

class UserInDB(UserBase):
    hashed_password: str
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True
        
class UserRead(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
