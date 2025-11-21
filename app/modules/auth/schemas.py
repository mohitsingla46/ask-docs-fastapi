from typing import Annotated, Optional
from pydantic import BeforeValidator, BaseModel, Field

PyObjectId = Annotated[str, BeforeValidator(str)]


class UserBase(BaseModel):
    sub: str
    email: str
    name: str
    picture: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class UserLogin(BaseModel):
    token: str


class User(UserBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class UserLoginResponse(UserBase):
    access_token: str


class UserCreate(UserBase):
    pass
