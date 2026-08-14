from fastapi import APIRouter, Depends

from app.auth.dependencies import Principal, get_current_user

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("")
def list_papers(principal: Principal = Depends(get_current_user)):
    return []
