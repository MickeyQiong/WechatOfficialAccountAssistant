# ──────────────────────────────────────────────────────────────────
# 微信公众号文章发表助手 — Docker 镜像
# 
# 基于 CrewAI Flow 的多 Agent 协作系统，自动完成公众号文章
# 的内容创作（规划→搜索→撰写）和发布处理（插图→排版→审校）
# ──────────────────────────────────────────────────────────────────

# ===== Stage 1: 安装依赖 =====
FROM python:3.12-slim-bookworm AS builder

# 安装 uv（快速 Python 包管理器）
COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 先复制依赖描述文件（利用 Docker 缓存层）
COPY pyproject.toml uv.lock ./

# 创建虚拟环境并安装依赖（不安装项目本身，仅安装依赖）
RUN uv sync --frozen --no-install-project

# ===== Stage 2: 运行镜像 =====
FROM python:3.12-slim-bookworm AS runner

# 安装系统级依赖（部分 Python 包需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制 uv 工具链
COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

# 复制虚拟环境（包含所有已安装依赖）
COPY --from=builder /app/.venv /app/.venv

# 复制项目源代码和配置
COPY pyproject.toml ./
COPY src/ /app/src/
COPY knowledge/ /app/knowledge/

# 设置 Python 路径，使虚拟环境和 src 目录都可被导入
# 项目使用 src layout，源码在 /app/src/assistant_flow/
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# ── 环境变量（运行时必须提供）──────────────────────────────────
# 以下变量需要在 docker run 时通过 -e 或 --env-file 传入：
#
#   OPENAI_API_KEY        OpenAI API 密钥（必填）
#   OPENAI_BASE_URL       OpenAI 兼容 API 地址（可选，默认 OpenAI 官方）
#   WECHAT_APP_ID         微信公众号 AppID（发布草稿时必填）
#   WECHAT_APP_SECRET     微信公众号 AppSecret（发布草稿时必填）
#   WECHAT_ARTICLES_DIR   往期文章目录（可选，默认 knowledge/articles/）
#   MODEL                 默认模型（可选，如 gpt-4o）
# ────────────────────────────────────────────────────────────────

# 默认 API 基础 URL（可通过环境变量覆盖）
ENV OPENAI_BASE_URL=""

# 创建输出目录
RUN mkdir -p /app/output

# 默认入口：执行完整流程
ENTRYPOINT ["uv", "run"]
CMD ["kickoff"]
