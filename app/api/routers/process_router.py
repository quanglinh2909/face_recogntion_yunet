import time

from fastapi import APIRouter



router = APIRouter()

@router.post("/enable")
async def start_process(req):
    return {"message": "Process enabled"}

@router.post("/kill")
async def stop_process(req):
    return {"message": "Process killed"}
