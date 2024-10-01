import threading
from fastapi import FastAPI
import os
from contextlib import asynccontextmanager

from app.api.main import api_router
from app.core.config import settings
from app.api.routers.websocket_routes import router as api_router_ws
from utils.main import start_detect


@asynccontextmanager
async def lifespan(app: FastAPI):
    # PikaPublisher()
    threading.Thread(target=start_detect, args=(), daemon=True).start()
    print("Starting the server")
    yield
    print("Shutting down the server")

    # os.system("rm -rf temp")


app = FastAPI(
    title="Oryza AI FastAPI Backend",
    docs_url="/",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="")
app.include_router(api_router_ws, prefix="/ws")
