import uvicorn

from app.core.config import settings
if __name__ == "__main__":
    if settings.ENVIRONMENT == "dev":
        uvicorn.run("app.main:app", host="0.0.0.0", reload=True, port=settings.PORT)
    else:
        uvicorn.run("app.main:app", host="0.0.0.0", reload=False,port=settings.PORT)
