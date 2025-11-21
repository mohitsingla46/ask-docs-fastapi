from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(
        self,
        model: Type[ModelType],
        db_client: AsyncIOMotorClient,
        database_name: str,
        collection_name: str,
    ):
        self.model = model
        self.db_client = db_client
        self.database_name = database_name
        self.collection_name = collection_name

    @property
    def collection(self):
        return self.db_client[self.database_name][self.collection_name]

    async def get(self, id: str) -> Optional[ModelType]:
        doc = await self.collection.find_one({"_id": ObjectId(id)})
        if doc:
            return self.model(**doc)
        return None

    async def get_multi(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        cursor = self.collection.find().skip(skip).limit(limit)
        items = await cursor.to_list(length=limit)
        return [self.model(**item) for item in items]

    async def create(self, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = jsonable_encoder(obj_in)
        result = await self.collection.insert_one(obj_in_data)
        created_doc = await self.collection.find_one({"_id": result.inserted_id})
        return self.model(**created_doc)

    async def update(
        self, id: str, obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> Optional[ModelType]:
        obj_data = jsonable_encoder(obj_in)
        update_result = await self.collection.update_one(
            {"_id": ObjectId(id)}, {"$set": obj_data}
        )
        if update_result.modified_count > 0:
            return await self.get(id)
        return None

    async def remove(self, id: str) -> bool:
        result = await self.collection.delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0
