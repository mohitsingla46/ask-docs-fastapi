from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.modules.document import schemas
from app.modules.document.service import DocumentService
from app.modules.document import dependencies as deps
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.schemas import User

router = APIRouter()

@router.post('/upload', response_model=schemas.Document)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService  = Depends(deps.get_document_service)
):
    return await document_service.upload_document(file, current_user.id)

@router.get('/fetch', response_model=schemas.Document)
async def get_document(
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(deps.get_document_service)
):
    document = await document_service.get_document(current_user.id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document