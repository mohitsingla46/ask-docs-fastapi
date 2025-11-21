from fastapi import APIRouter, Depends
from app.modules.auth import schemas
from app.modules.auth.service import AuthService
from app.modules.auth import dependencies as deps

router = APIRouter()

@router.post("/google", response_model=schemas.UserLoginResponse)
async def googleLogin(
    payload: schemas.UserLogin,
    auth_service: AuthService = Depends(deps.get_auth_service)
):
    return await auth_service.google_login(payload.token)
