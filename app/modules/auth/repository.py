from app.repositories.base import BaseRepository
from app.modules.auth.schemas import User, UserCreate

class AuthRepository(BaseRepository[User, UserCreate, UserCreate]):
    async def get_by_sub(self, sub: str) -> User | None:
        """Get user by Google sub (subject) ID"""
        doc = await self.collection.find_one({"sub": sub})
        if doc:
            return self.model(**doc)
        return None

    async def update_by_sub(self, sub: str, update_data: dict) -> bool:
        """Update user by Google sub (subject) ID"""
        result = await self.collection.update_one(
            {"sub": sub},
            {"$set": update_data}
        )
        return result.modified_count > 0