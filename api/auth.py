"""登录与当前用户接口。"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from services.auth.dependencies import (
    create_access_token,
    fetch_user_by_username,
    get_current_user,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)


@router.post("/login")
def login(body: LoginBody, request: Request):
    row = fetch_user_by_username(request.app.state.engine, body.username)
    if not row or not row["is_active"] or not verify_password(
        body.password, row["password_hash"]
    ):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(row["username"], row["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": row["username"],
        "role": row["role"],
    }


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}
