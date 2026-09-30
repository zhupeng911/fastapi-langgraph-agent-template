"""API v1 主路由配置：引入认证、聊天机器人等不同功能的子路由."""

from app.api.v1.auth import auth_router
from fastapi import APIRouter

# 注册子路由
api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["Auth"])
