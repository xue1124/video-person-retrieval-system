import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import text

from db import load_siglip_env

load_siglip_env()
SECRET_KEY = os.environ.get("JWT_SECRET", "").strip()
if len(SECRET_KEY) < 32 or SECRET_KEY.startswith("CHANGE_ME"):
    raise RuntimeError("Set JWT_SECRET to a private random value of at least 32 characters.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def create_access_token(sub: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": sub, "role": role, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def fetch_user_by_username(engine, username: str):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, username, password_hash, role, is_active FROM users WHERE username=:u"
            ),
            {"u": username},
        ).mappings().first()
    return row


def _user_from_token_string(engine, token_str: str) -> dict:
    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或令牌缺失",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(token_str)
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="无效令牌")
    except JWTError:
        raise HTTPException(status_code=401, detail="令牌无效或已过期")

    row = fetch_user_by_username(engine, username)
    if not row or not row["is_active"]:
        raise HTTPException(status_code=401, detail="用户不存在或已禁用")
    return dict(row)


async def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或令牌缺失",
            headers={"WWW-Authenticate": "Bearer"},
        )
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=500, detail="数据库未初始化")
    return _user_from_token_string(engine, creds.credentials)


async def get_current_user_media(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Query(None, description="供 <video> 标签使用的 JWT（与 Authorization 二选一）"),
):
    """媒体文件播放：支持 Header Bearer 或 query token。"""
    token_str = (creds.credentials if creds and creds.credentials else None) or token
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=500, detail="数据库未初始化")
    return _user_from_token_string(engine, token_str or "")


async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
