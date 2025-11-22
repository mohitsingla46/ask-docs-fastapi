from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from app.modules.auth.schemas import User
from app.modules.auth.service import AuthService
from app.core.config import settings
from motor.motor_asyncio import AsyncIOMotorClient
from app.modules.auth.repository import AuthRepository
from app.db.mongodb import get_database


def get_auth_repository(
    db: AsyncIOMotorClient = Depends(get_database),
) -> AuthRepository:
    return AuthRepository(User, db, settings.DATABASE_NAME, "users")


def get_auth_service(
    auth_repository: AuthRepository = Depends(get_auth_repository),
) -> AuthService:
    return AuthService(auth_repository)


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    try:
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET, algorithms=["HS256"])
        sub: str = payload.get("sub")
        if sub is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = await auth_service.auth_repository.get_by_sub(sub)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
