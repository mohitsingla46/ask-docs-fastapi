from fastapi import APIRouter, Depends
from app.modules.agent import schemas
from app.modules.auth.dependencies import get_current_user
from app.modules.agent.service import AgentService
from app.modules.agent import dependencies as deps
from app.modules.auth.schemas import User

router = APIRouter()

@router.post("/chat", response_model=schemas.ChatResponse)
async def chat_with_agent(
    payload: schemas.ChatInput,
    current_user: User = Depends(get_current_user),
    agent_service: AgentService  = Depends(deps.get_agent_service)
):
    chat = await agent_service.handle_chat(
        user_id=current_user.id,
        thread_id=payload.thread_id,
        message=payload.message
    )
    return {"chat": chat}