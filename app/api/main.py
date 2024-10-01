from fastapi import APIRouter
from app.api.routers import (process_router)

api_router = APIRouter()
api_router.include_router(process_router.router, prefix="", tags=["process"])