from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient
from app.db.mongodb import get_database
from app.modules.document.repository import DocumentRepository
from app.modules.document.service import DocumentService
from app.modules.document.schemas import Document
from app.core.config import settings

def get_document_repository(db: AsyncIOMotorClient = Depends(get_database)) -> DocumentRepository:
    return DocumentRepository(Document, db, settings.DATABASE_NAME, "documents")

def get_document_service(document_repository: DocumentRepository = Depends(get_document_repository)) -> DocumentService:
    return DocumentService(document_repository)
