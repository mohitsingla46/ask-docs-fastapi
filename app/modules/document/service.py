from app.modules.document.repository import DocumentRepository
from fastapi import UploadFile
from app.modules.document.schemas import Document, DocumentCreate
import os
from app.core.config import settings
import shutil
from app.modules.document.ingestion import process_document, clear_user_vectors

class DocumentService:
    def __init__(self, document_repository: DocumentRepository):
        self.document_repository = document_repository

    async def upload_document(self, file: UploadFile, user_id: str) -> Document:
        existing = await self.document_repository.get_by_user_id(user_id)
        
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        
        content = await file.read()
        await file.seek(0)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            process_document(file_path, user_id)
        except Exception as e:
            print(f"Error processing document: {e}")
        
        if existing:
            old_path = existing.path
            update_data = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": os.path.getsize(file_path),
                "content": content.decode('utf-8'),
                "path": file_path,
                "user_id": user_id
            }
            updated_doc = await self.document_repository.update(existing.id, update_data)
            if old_path != file_path and os.path.exists(old_path):
                os.remove(old_path)
            return updated_doc
        else:
            document_in = DocumentCreate(
                filename=file.filename,
                content_type=file.content_type,
                size=os.path.getsize(file_path),
                content=content.decode('utf-8'),
                path=file_path,
                user_id=user_id
            )
            return await self.document_repository.create(document_in)
    
    async def get_document(self, user_id: str) -> Document:
        document = await self.document_repository.get_by_user_id(user_id)
        if not document:
            return None
        return document
    
    async def delete_document(self, user_id: str) -> None:
        document = await self.document_repository.get_by_user_id(user_id)
        if document:
            print(f"Deleting document at path: {document.id}")
            if os.path.exists(document.path):
                os.remove(document.path)
            await self.document_repository.delete(document.id)
            try:
                clear_user_vectors(user_id)
            except Exception as e:
                print(f"Error clearing user vectors: {e}")

