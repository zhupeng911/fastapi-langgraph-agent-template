"""应用数据库服务.

统一提供用户、会话和消息的全部数据库CRUD操作，类似于java mapper.

"""

from typing import (
    List,
    Optional,
)

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import (
    Session,
    col,
    create_engine,
    select,
)

from app.core.config import (
    settings,
)
from app.core.logging import logger
from app.models.session import Session as ChatSession
from app.models.user import User


class DatabaseService:
    """数据库操作服务类.

    该类负责用户、会话和消息的全部数据库操作，使用 SQLModel 进行 ORM 操作并维护连接池.
    """

    def __init__(self):
        """使用连接池初始化数据库服务."""
        try:
            # 配置数据库连接池参数
            pool_size = settings.POSTGRES_POOL_SIZE
            max_overflow = settings.POSTGRES_MAX_OVERFLOW
            connection_url = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )

            # 创建数据库引擎并启用连接池
            self.engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=30,
                pool_recycle=1800,
            )
        except SQLAlchemyError as e:
            logger.error("database_initialization_error", error=str(e), environment=settings.ENVIRONMENT.value)

    # 创建用户
    async def create_user(self, email: str, password: str, username: str | None = None) -> User:
        """创建新用户.

        参数：
            email: 用户邮箱地址.
            password: 已哈希的密码.
            username: 可选的显示名称.

        返回：
            User: 创建的用户.
        """
        with Session(self.engine) as session:
            user = User(email=email, hashed_password=password, username=username)
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    # 根据 ID 获取用户
    async def get_user(self, user_id: int) -> Optional[User]:
        """根据 ID 获取用户.

        参数：
            user_id: 要查询的用户 ID.

        返回：
            Optional[User]: 查询到的用户；不存在时返回 None.
        """
        with Session(self.engine) as session:
            user = session.get(User, user_id)
            return user

    # 根据邮箱获取用户
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户.

        参数：
            email: 要查询的用户邮箱.

        返回：
            Optional[User]: 查询到的用户；不存在时返回 None.
        """
        with Session(self.engine) as session:
            statement = select(User).where(User.email == email)
            user = session.exec(statement).first()
            return user

    # 根据邮箱删除用户
    async def delete_user_by_email(self, email: str) -> bool:
        """根据邮箱删除用户.

        参数：
            email: 要删除的用户邮箱.

        返回：
            bool: 删除成功返回 True，用户不存在返回 False.
        """
        with Session(self.engine) as session:
            user = session.exec(select(User).where(User.email == email)).first()
            if not user:
                return False

            session.delete(user)
            session.commit()
            return True

    # 创建会话
    async def create_session(
        self, session_id: str, user_id: int, name: str = "", username: str | None = None
    ) -> ChatSession:
        """创建新的聊天会话.

        参数：
            session_id: 新会话 ID.
            user_id: 会话所属用户的 ID.
            name: 可选的会话名称，默认为空字符串.
            username: 从用户复制的显示名称，用于个性化 LLM 交互.

        返回：
            ChatSession: 创建的会话.
        """
        with Session(self.engine) as session:
            chat_session = ChatSession(id=session_id, user_id=user_id, name=name, username=username)
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            return chat_session

    # 删除会话
    async def delete_session(self, session_id: str) -> bool:
        """根据 ID 删除会话.

        参数：
            session_id: 要删除的会话 ID.

        返回：
            bool: 删除成功返回 True，会话不存在返回 False.
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                return False

            session.delete(chat_session)
            session.commit()
            return True

    # 根据 ID 获取会话
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """根据 ID 获取会话.

        参数：
            session_id: 要查询的会话 ID.

        返回：
            Optional[ChatSession]: 查询到的会话；不存在时返回 None.
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            return chat_session

    # 获取用户会话
    async def get_user_sessions(self, user_id: int) -> List[ChatSession]:
        """获取用户的全部会话.

        参数：
            user_id: 用户 ID.

        返回：
            List[ChatSession]: 用户的会话列表.
        """
        with Session(self.engine) as session:
            statement = (
                select(ChatSession).where(col(ChatSession.user_id) == user_id).order_by(col(ChatSession.created_at))
            )
            sessions = session.exec(statement).all()
            return list(sessions)

    # 更新会话名称
    async def update_session_name(self, session_id: str, name: str) -> ChatSession:
        """更新会话名称.

        参数：
            session_id: 要更新的会话 ID.
            name: 会话的新名称.

        返回：
            ChatSession: 更新后的会话.

        异常：
            HTTPException: 会话不存在时抛出.
        """
        with Session(self.engine) as session:
            chat_session = session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")

            chat_session.name = name
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            return chat_session

    # 获取会话工厂
    def get_session_maker(self):
        """获取用于创建数据库会话的会话对象.

        返回：
            Session: SQLModel 会话对象.
        """
        return Session(self.engine)

    # 健康检查
    async def health_check(self) -> bool:
        """检查数据库连接健康状态.

        返回：
            bool: 数据库健康返回 True，否则返回 False.
        """
        try:
            with Session(self.engine) as session:
                # 执行简单查询检查连接
                session.exec(select(1)).first()
                return True
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False


# 创建单例实例
database_service = DatabaseService()
