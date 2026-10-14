"""
API Token 鉴权模块
支持 Bearer Token 方式验证 API 请求
"""

import os
import secrets
import hmac
from fastapi import Header, HTTPException
from typing import Optional

# 从 .env 读取或自动生成 API Token
_API_TOKEN: Optional[str] = None


def get_api_token() -> str:
    """获取当前 API Token，不存在则从 .env 读取或自动生成"""
    global _API_TOKEN
    if _API_TOKEN is None:
        _API_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()
        if not _API_TOKEN:
            _API_TOKEN = secrets.token_hex(32)
            # 尝试写入 .env
            try:
                env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
                with open(env_path, "a", encoding="utf-8") as f:
                    f.write(f'\nAPI_BEARER_TOKEN={_API_TOKEN}\n')
                print("🔑 已自动生成 API Token，并写入 .env")
            except Exception:
                print("🔑 API Token 已生成，但无法写入 .env；请手动配置 API_BEARER_TOKEN")
    return _API_TOKEN


async def verify_token(authorization: Optional[str] = Header(None)):
    """FastAPI 依赖注入：验证 Bearer Token"""
    token = get_api_token()
    if not token:
        # 未配置 Token 则不鉴权
        return True

    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")

    scheme, _, cred = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not cred or not hmac.compare_digest(cred, token):
        raise HTTPException(status_code=401, detail="无效的 API Token")

    return True
