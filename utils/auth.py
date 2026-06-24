"""
API Token 鉴权模块
支持 Bearer Token 方式验证 API 请求
"""

import os
import secrets
from fastapi import Header, HTTPException
from typing import Optional

# 从 .env 读取或自动生成 API Token
_API_TOKEN: Optional[str] = None


def get_api_token() -> str:
    """获取当前 API Token，不存在则从 .env 读取或自动生成"""
    global _API_TOKEN
    if _API_TOKEN is None:
        _API_TOKEN = os.getenv("API_BEARER_TOKEN", "")
        if not _API_TOKEN:
            _API_TOKEN = secrets.token_hex(32)
            # 尝试写入 .env
            try:
                env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
                with open(env_path, "a", encoding="utf-8") as f:
                    f.write(f'\nAPI_BEARER_TOKEN={_API_TOKEN}\n')
                print(f"🔑 已自动生成 API Token，已追加到 .env 文件")
            except Exception:
                print(f"🔑 自动生成的 API Token: {_API_TOKEN}")
    return _API_TOKEN


async def verify_token(authorization: Optional[str] = Header(None)):
    """FastAPI 依赖注入：验证 Bearer Token"""
    token = get_api_token()
    if not token:
        # 未配置 Token 则不鉴权
        return True

    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")

    scheme, _, cred = authorization.partition(" ")
    if scheme.lower() != "bearer" or cred != token:
        raise HTTPException(status_code=401, detail="无效的 API Token")

    return True
