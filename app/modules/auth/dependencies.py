from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient
from app.modules.auth.repository import AuthRepository
from app.modules.auth.service import AuthService
from app.core.config import settings
from app.db.mongodb import get_database
from app.modules.auth.schemas import User


def get_auth_repository(
    db: AsyncIOMotorClient = Depends(get_database),
) -> AuthRepository:
    return AuthRepository(User, db, settings.DATABASE_NAME, "users")


def get_auth_service(
    auth_repository: AuthRepository = Depends(get_auth_repository),
) -> AuthService:
    return AuthService(auth_repository)
