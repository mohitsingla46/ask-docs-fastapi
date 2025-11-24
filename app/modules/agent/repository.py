from app.repositories.base import BaseRepository
from app.modules.agent.schemas import Chat, ChatCreate
from typing import List
from datetime import datetime

class AgentRepository(BaseRepository[Chat, ChatCreate, ChatCreate]):
    async def get_chats_by_thread(self, user_id: str, thread_id: str) -> List[Chat]:
        """Get all chats for a specific user and thread, ordered by creation time."""
        cursor = self.collection.find(
            {"user_id": user_id, "thread_id": thread_id}
        ).sort("created_at", 1)
        items = await cursor.to_list(length=None)
        return [self.model(**item) for item in items]

    async def create_chat(self, user_id: str, thread_id: str, message_type: str, message: str) -> Chat:
        """Create a new chat message."""
        chat_data = ChatCreate(
            user_id=user_id,
            thread_id=thread_id,
            type=message_type,
            message=message,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        return await self.create(chat_data)
    
    async def delete_chats_by_thread(self, user_id: str, thread_id: str) -> int:
        """Delete all chats for a specific user and thread."""
        result = await self.collection.delete_many(
            {"user_id": user_id, "thread_id": thread_id}
        )
        return result.deleted_count