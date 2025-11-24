from typing import Annotated, Optional
from pydantic import BeforeValidator, BaseModel, Field

PyObjectId = Annotated[str, BeforeValidator(str)]

class ChatBase(BaseModel):
    user_id: str
    thread_id: str
    type: str
    message: str
    created_at: str | None = None
    updated_at: str | None = None

class ChatCreate(ChatBase):
    pass

class Chat(ChatBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ChatResponse(BaseModel):
    chat: Chat

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True

class ChatInput(BaseModel):
    thread_id: str
    message: str