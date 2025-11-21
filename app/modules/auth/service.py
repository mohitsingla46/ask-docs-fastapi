from app.modules.auth.repository import AuthRepository
from app.modules.auth.schemas import UserLoginResponse, UserBase, User, UserCreate
from app.core.config import settings
from datetime import datetime, timedelta
from jose import JWTError, jwt
from google.auth.transport import requests
from google.oauth2 import id_token
import google.auth.exceptions


class AuthService:
    def __init__(self, auth_repository: AuthRepository):
        self.auth_repository = auth_repository

    async def google_login(self, token: str) -> UserLoginResponse:
        try:
            id_info = id_token.verify_oauth2_token(
                token,
                requests.Request(),
                settings.GOOGLE_CLIENT_ID
            )

            user_data = UserBase(
                sub=id_info['sub'],
                email=id_info['email'],
                name=id_info.get('name', ''),
                picture=id_info.get('picture')
            )

            existing_user = await self.auth_repository.get_by_sub(user_data.sub)
            if not existing_user:
                user_create = UserCreate(
                    sub=user_data.sub,
                    email=user_data.email,
                    name=user_data.name,
                    picture=user_data.picture,
                    created_at=datetime.utcnow().isoformat(),
                    updated_at=datetime.utcnow().isoformat()
                )
                created_user = await self.auth_repository.create(user_create)
                user_data.created_at = created_user.created_at
                user_data.updated_at = created_user.updated_at
            else:
                user_data.created_at = existing_user.created_at
                user_data.updated_at = datetime.utcnow().isoformat()
                await self.auth_repository.update_by_sub(user_data.sub, {"updated_at": user_data.updated_at})

            access_token = self._create_access_token({"sub": user_data.sub, "email": user_data.email})

            return UserLoginResponse(
                sub=user_data.sub,
                email=user_data.email,
                name=user_data.name,
                picture=user_data.picture,
                created_at=user_data.created_at,
                updated_at=user_data.updated_at,
                access_token=access_token
            )

        except google.auth.exceptions.GoogleAuthError as e:
            raise ValueError(f"Invalid Google token: {str(e)}")
        except Exception as e:
            raise ValueError(f"Authentication failed: {str(e)}")

    def _create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days expiration
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")
        return encoded_jwt
