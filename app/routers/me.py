from fastapi import APIRouter, Depends

from app.auth.dependencies import Principal, get_current_user

router = APIRouter(prefix="/me", tags=["me"])


@router.get("")
def read_me(principal: Principal = Depends(get_current_user)):
    return {"uid": principal.uid, "email": principal.email}
