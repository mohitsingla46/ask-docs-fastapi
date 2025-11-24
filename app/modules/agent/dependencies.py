from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient
from app.db.mongodb import get_database
from app.modules.agent.repository import AgentRepository
from app.modules.agent.service import AgentService
from app.modules.agent.schemas import Chat
from app.core.config import settings


def get_agent_repository(
    db: AsyncIOMotorClient = Depends(get_database),
) -> AgentRepository:
    return AgentRepository(Chat, db, settings.DATABASE_NAME, "chats")


def get_agent_service(
    document_repository: AgentRepository = Depends(get_agent_repository),
) -> AgentService:
    return AgentService(document_repository)
