"""本地启动 FastAPI 的统一入口。"""
from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8002")),
        reload=False,
    )
