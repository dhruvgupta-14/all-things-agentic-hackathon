from fastapi import APIRouter, Depends

from app.auth.dependencies import Principal, get_current_user

router = APIRouter(prefix="/me", tags=["me"])


@router.get("")
async def read_me(principal: Principal = Depends(get_current_user)):
    return {
        "user_id": str(principal.user_id),
        "auth_subject": principal.auth_subject,
        "email": principal.email,
    }
