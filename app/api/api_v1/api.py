from fastapi import APIRouter
from app.modules.auth import controller as auth
from app.modules.document import controller as document
from app.modules.agent import controller as agent

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(document.router, prefix="/documents", tags=["documents"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])