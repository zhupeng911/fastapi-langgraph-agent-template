"""共用基础模型."""

from datetime import datetime, UTC

from sqlmodel import Field, SQLModel


# SQLModel基类，基类不要声明table=True，因为没有主键会报错：Mapper[BaseModel(basemodel)] could not assemble any primary key columns
# 1.SQLModel基类既是一个 Pydantic 模型（支持数据校验），也是一个 SQLAlchemy ORM 模型（映射到数据库表）
# 2.Field 用于定义模型字段的元数据和约束条件，指定主键（primary_key=True）、建立索引（index=True）、设置默认值（default=）等
class BaseModel(SQLModel):
    """包含公共字段的基础模型."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
