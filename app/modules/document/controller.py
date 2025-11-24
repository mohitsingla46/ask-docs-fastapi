from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from app.modules.document import schemas
from app.modules.document.service import DocumentService
from app.modules.document import dependencies as deps
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.schemas import User

router = APIRouter()

@router.post('/upload')
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    document_service: DocumentService  = Depends(deps.get_document_service)
):
    document = await document_service.upload_document(file, current_user.id)
    return {"document": document}

@router.get('/fetch')
async def get_document(
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(deps.get_document_service)
):
    document = await document_service.get_document(current_user.id)
    return {"document": document}

@router.delete('/delete', status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    current_user: User = Depends(get_current_user),
    document_service: DocumentService = Depends(deps.get_document_service)
):
    await document_service.delete_document(current_user.id)
    return None