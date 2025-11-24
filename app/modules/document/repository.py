from app.repositories.base import BaseRepository
from app.modules.document.schemas import Document, DocumentCreate
from bson import ObjectId

class DocumentRepository(BaseRepository[Document, DocumentCreate, DocumentCreate]):
    async def get_by_user_id(self, user_id: str) -> Document | None:
        doc = await self.collection.find_one({"user_id": user_id})
        if doc:
            return self.model(**doc)
        return None
    
    async def delete(self, document_id: str) -> None:
        await self.collection.delete_one({"_id": ObjectId(document_id)})