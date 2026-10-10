# FastAPI LangGraph Agent 模板
一个用于使用 FastAPI 和 LangGraph 构建 AI Agent 后端的生产级模板。它处理有状态对话、长期记忆、工具调用、可观测性、限流和身份认证等复杂部分，让你可以专注于 Agent 逻辑。

**面向需要可靠基础设施的 AI 工程师**，而不是一个教程项目。



**由 Atlas Cloud 驱动——可直接替换的 LangGraph Agent LLM 后端**

[**Atlas Cloud**](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=fastapi-langgraph-agent-production-ready-template) 提供与 **OpenAI 兼容的 LLM API**，可以无缝集成到这个 FastAPI + LangGraph 模板中，无需修改 Agent 图代码。只需替换 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，即可通过统一端点使用 **DeepSeek、Qwen、GLM、Kimi、MiniMax、Gemini、Claude、GPT** 等模型。

此模板中的 `LLMRegistry` 使用 `langchain_openai.ChatOpenAI`；Atlas Cloud 在接口层面兼容，因此无需改动任何 LangGraph 逻辑，即可立即访问 130 多个精选模型。

# 功能概览
+ **LangGraph** 有状态 Agent，支持检查点、工具调用和人在回路
+ 通过 mem0 + pgvector 实现的**长期记忆**——按用户进行语义搜索，并由缓存提供支持
+ **LLM 服务**支持循环模型回退、指数退避重试和总超时预算
+ 所有 LLM 调用均支持 **Langfuse** 追踪；提供 Prometheus 指标和 Grafana 仪表板
+ 支持带会话管理的 **JWT 身份认证**；通过 slowapi 实现限流
+ 支持 **Alembic** 数据库迁移；可选 Valkey/Redis 缓存层
+ 每行日志都包含请求、会话和用户上下文的**结构化日志**

# 项目结构
```plain
app/
  api/v1/          # Route handlers
  core/
    langgraph/     # Agent graph + tools
    prompts/       # System prompt template
    cache.py       # Valkey/Redis + in-memory fallback
    config.py      # Settings
    middleware.py  # Metrics, logging context, profiling
    limiter.py     # Rate limiting
  models/          # SQLModel ORM models
  schemas/         # Pydantic request/response schemas
  services/        # LLM, database, memory services
alembic/           # Database migrations
evals/             # LLM evaluation framework
```

# 快速开始
## 前置条件
+ Python 3.13+
+ [uv](https://docs.astral.sh/uv/)：`pip install uv`
+ Docker + Docker Compose（本地开发推荐）
+ OpenAI API 密钥
+ Langfuse 账户（可选；设置 `LANGFUSE_TRACING_ENABLED=false` 可跳过）

## 方案 A：Docker（推荐）
这是最快的启动方式.一条命令即可启动 API 和带 pgvector 的 PostgreSQL.

```bash
git clone <repo-url> my-agent
cd my-agent

# 复制并填写环境文件
cp .env.example .env.development
# 必填：OPENAI_API_KEY、JWT_SECRET_KEY
# 可选：LANGFUSE_* 密钥（或设置 LANGFUSE_TRACING_ENABLED=false）

make install       # 安装 Python 依赖和 pre-commit 钩子
make docker-up     # 启动 API（端口 8000）和 PostgreSQL
make docker-migrate # 在 app 容器中运行 Alembic 迁移
```

打开 [http://localhost:8000/docs](http://localhost:8000/docs).

## 方案 B：本地 Python
```bash
git clone <repo-url> my-agent
cd my-agent

cp .env.example .env.development
# 填写：OPENAI_API_KEY、JWT_SECRET_KEY、POSTGRES_*（指向你的数据库）

make install       # 安装依赖和 pre-commit 钩子
make migrate       # 通过 Alembic 创建表
make dev           # 在端口 8000 启动带热重载的服务
```

打开 [http://localhost:8000/docs](http://localhost:8000/docs) 查看交互式 API 文档。

# 常见问题
## 通用问题
**这个模板是什么？**  
这是一个基于 FastAPI + LangGraph 构建的 AI Agent 后端生产级基础模板。它集成了通常需要手动组装的组件，包括有状态对话、长期记忆、工具调用、可观测性、限流和 JWT 身份认证。

**它与基础 LangGraph 配置有什么不同？**  
基础 LangGraph 快速入门只实现了“Agent 在本地运行”。此模板额外提供 Alembic 迁移、mem0 + pgvector 长期记忆、Langfuse 追踪、Prometheus + Grafana 仪表板、JWT 会话、slowapi 限流、带请求级上下文的结构化日志，以及循环回退 LLM 服务，涵盖了通常需要单独构建的生产环境能力。

## 安装与配置
**必须使用 Docker 吗？**  
推荐使用，但不是必须的。`make docker-up` 会同时启动 API 和 PostgreSQL。仅进行本地配置时，请参阅 [docs/getting-started.md](docs/getting-started.md)。

**支持哪些 LLM 提供商？**  
目前通过 `app/services/llm/registry.py` 中的 `LLMRegistry` 仅支持 **OpenAI**。计划通过 LangChain 的 `init_chat_model` 支持多提供商（Anthropic、Google、OpenRouter），详见 [#51](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template/issues/51)。请通过 `.env.development` 中的 `DEFAULT_LLM_MODEL` 配置模型。

**如何配置长期记忆？**  
长期记忆采用自托管方式：mem0 在进程内运行，并通过 pgvector 持久化到现有 PostgreSQL 中，不需要单独的 mem0 云账号或 API 密钥。你只需要一个可用的 `OPENAI_API_KEY`（用于事实提取和向量嵌入），并启用 pgvector 扩展。详情请参阅 [docs/memory.md](docs/memory.md)。

## 开发
**如何添加自定义工具？**  
在 `app/core/langgraph/tools/` 中添加一个使用 LangChain `@tool` 装饰的函数，并将其注册到该包导出的 `tools` 列表中。Agent 下次启动时会自动加载，无需修改图结构。

**LLM 服务如何处理故障？**  
分为两层：(1) 通过 `tenacity` 实现每次调用的指数退避重试；(2) **循环回退**——如果当前模型耗尽重试次数，服务会切换到 `LLMRegistry` 中的下一个模型并继续处理。总超时预算会限制整个调用的耗时，从而控制延迟。详见 [docs/llm-service.md](docs/llm-service.md)。

**不使用 Langfuse 可以吗？**  
可以。将 `LANGFUSE_TRACING_ENABLED=false`（或不配置 Langfuse 相关密钥）即可。Agent 无需修改即可运行，结构化日志仍会记录请求、会话和用户上下文。

## 故障排查
**API 无法启动**

+ 确认 PostgreSQL 正在运行（`make docker-up` 会与 API 一起启动 PostgreSQL）
+ 确认 `.env.development` 存在——从 `.env.example` 复制并填写必需的配置项
+ 执行迁移：`make migrate`

**记忆或语义搜索没有返回结果**

+ 确认 PostgreSQL 实例已启用 `pgvector` 扩展
+ 确认 `OPENAI_API_KEY` 有效（mem0 会调用 OpenAI 进行事实提取和向量嵌入）
+ 检查 `.env.development` 中是否设置了 `LONG_TERM_MEMORY_MODEL` 和 `LONG_TERM_MEMORY_EMBEDDER_MODEL`

**限流过于严格**  
限制定义在 `app/core/limiter.py`（slowapi）中。可以在该文件中调整各路由装饰器或默认限流值。相关环境变量请参阅 [docs/configuration.md](docs/configuration.md)。

# 贡献
欢迎提交 PR。请先阅读 [docs/getting-started.md](docs/getting-started.md) 配置开发环境，然后遵循 [AGENTS.md](AGENTS.md) 中的编码规范。

请私下报告安全问题，详见 [SECURITY.md](SECURITY.md)。

