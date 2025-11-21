from fastapi import FastAPI
from dotenv import load_dotenv
from app.core.config import settings
from starlette.middleware.cors import CORSMiddleware
from app.api.api_v1.api import api_router
from app.db.mongodb import db
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

@app.on_event("startup")
async def startup_db_client():
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)
    print("Connected to MongoDB")

@app.on_event("shutdown")
async def shutdown_db_client():
    db.client.close()
    print("Disconnected from MongoDB")

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
