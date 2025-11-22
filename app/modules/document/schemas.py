from typing import Annotated, Optional
from pydantic import BeforeValidator, BaseModel, Field
from datetime import datetime

PyObjectId = Annotated[str, BeforeValidator(str)]

class DocumentBase(BaseModel):
    filename: str
    content: str
    content_type: str
    size: int
    user_id: str

class DocumentCreate(DocumentBase):
    path: str

class Document(DocumentBase):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True