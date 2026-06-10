# 📱 微信公众号文章发表助手

> 基于 [CrewAI](https://crewai.com) 多智能体框架的微信公众号文章自动生成与发布系统。输入一个主题，AI 智能体团队将自动完成内容规划、风格分析、网络搜索、文章撰写、配图设计、排版优化、审校合规，最终产出可直接发布的微信文章。

[![Python](https://img.shields.io/badge/Python-≥3.10%2C<3.14-blue)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-≥1.14.6-orange)](https://crewai.com)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

## ✨ 核心特点

- 🤖 **多智能体协作**：10+ 个专业 AI Agent 分工协作，模拟真实内容团队工作流程
- 📝 **智能内容创作**：从选题规划 → 风格参考 → 网络搜索 → 文章撰写，全自动完成
- 🎨 **配图与排版**：自动生成封面图方案、内容插图设计，并提供微信排版优化
- ✅ **质量保障**：内置审校校对和合规审查 Agent，确保文章质量与平台合规
- 📚 **往期文章学习**：基于 LlamaIndex 向量检索，分析历史文章风格，保持品牌调性一致
- 🔗 **智能内部链接**：自动发现相关往期文章，生成「相关阅读」板块
- 📤 **一键发布**：支持生成 Markdown 草稿并推送至微信公众号草稿箱

---

## 🏗️ 系统架构

```
用户输入标题
      │
      ▼
┌─────────────────────────────────────────────┐
│         WechatArticleFlow (主流程)            │
├─────────────────────────────────────────────┤
│                                             │
│  Step 1: ContentCrew（内容创作团队）           │
│  ┌─────────────────────────────────────┐    │
│  │ 内容规划师 → 风格分析师 → 网络搜索员 → 撰写者 │    │
│  └─────────────────────────────────────┘    │
│           │                                  │
│           ▼                                  │
│  Step 2: PublishingCrew（发布团队）           │
│  ┌─────────────────────────────────────┐    │
│  │ 插画设计师 → 排版设计师 → 审核校对员         │    │
│  │       → 合规审查员 → 草稿发布员        │    │
│  └─────────────────────────────────────┘    │
│           │                                  │
│           ▼                                  │
│  Step 3: 保存 → output/wechat_article.md     │
│                                             │
└─────────────────────────────────────────────┘
```

### Agent 角色矩阵

| Crew               | Agent 角色 | 职责                                 |
| ------------------ | ---------- | ------------------------------------ |
| **ContentCrew**    | 内容规划师 | 分析选题、设计大纲、优化标题         |
|                    | 风格分析师 | 学习往期文章风格，输出「风格指南卡」 |
|                    | 网络搜索员 | 搜索最新资讯、数据、案例             |
|                    | 配图生成师 | 设计封面图与插图方案                 |
|                    | 文章撰写者 | 撰写完整文章、添加内部链接           |
| **PublishingCrew** | 插画设计师 | 设计封面图和内容插图                 |
|                    | 排版设计师 | 微信 Markdown 排版优化               |
|                    | 审核校对员 | 文字校对、逻辑审核                   |
|                    | 合规审查员 | 微信平台规范检查                     |
|                    | 草稿发布员 | 生成最终草稿并保存/推送              |

---

## 📋 环境要求

| 依赖             | 版本/说明                                           |
| ---------------- | --------------------------------------------------- |
| Python           | `>=3.10, <3.14`                                     |
| UV               | 包管理器，[安装指南](https://docs.astral.sh/uv/)    |
| OpenAI API       | 需要`OPENAI_API_KEY`                                |
| 微信接口（可选） | 发布到草稿箱需要`WECHAT_APPID` + `WECHAT_APPSECRET` |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd WechatOfficialAccountAssistant
```

### 2. 安装依赖

```bash
# 安装 UV（如未安装）
pip install uv

# 进入项目目录并安装依赖
cd assistant_flow
uv sync
```

### 3. 配置环境变量

在 `assistant_flow/` 目录下创建 `.env` 文件：

```bash
# 必填
OPENAI_API_KEY=sk-your-api-key-here

# 可选：自定义模型
MODEL=gpt-4o

# 可选：微信草稿发布（需认证服务号）
WECHAT_APPID=wx_your_app_id
WECHAT_APPSECRET=your_app_secret
WECHAT_CONTENT_SOURCE_URL=https://your-site.com
```

### 4. 运行

```bash
cd assistant_flow
uv run kickoff
```

运行后，默认以 **「AI Agent 如何改变企业工作方式」** 为标题生成文章，输出到 `output/wechat_article.md`。

---

## 📖 使用方式

### 方式一：命令行运行（使用默认标题）

```bash
cd assistant_flow
uv run kickoff
```

### 方式二：通过 trigger payload 指定标题

```bash
cd assistant_flow
uv run run_with_trigger '{"title": "你的文章标题"}'
```

### 方式三：生成流程图可视化

```bash
cd assistant_flow
uv run plot
# 生成 wechat_article_flow.html 流程图
```

### 方式四：在代码中集成

```python
from assistant_flow.main import WechatArticleFlow

flow = WechatArticleFlow()
result = flow.kickoff({
    "crewai_trigger_payload": {"title": "2026年AI发展趋势"}
})

print(result)  # 最终文章内容
print(flow.state.final_article)  # 结构化访问
```

---

## 🐳 Docker 部署

### 构建镜像

```bash
cd assistant_flow
docker build -t wechat-article-assistant .
```

### 运行容器

**使用 `.env` 文件注入环境变量（推荐）：**

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  wechat-article-assistant
```

**手动指定环境变量：**

```bash
docker run --rm \
  -e OPENAI_API_KEY="sk-xxx" \
  -e OPENAI_BASE_URL="https://api.wlai.vip/v1" \
  -e WECHAT_APP_ID="wx4f..." \
  -e WECHAT_APP_SECRET="abc..." \
  -v $(pwd)/output:/app/output \
  wechat-article-assistant
```

> 💡 挂载 `output` 目录可将生成的文章持久化到宿主机。

### 指定文章标题

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  wechat-article-assistant uv run run_with_trigger '{"title": "你的文章标题"}'
```

### 生成流程图

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/output:/app/output \
  wechat-article-assistant uv run plot
```

### Docker Compose（可选）

创建 `docker-compose.yml`：

```yaml
services:
  wechat-assistant:
    build: .
    env_file:
      - .env
    volumes:
      - ./output:/app/output
      - ./knowledge:/app/knowledge
```

运行：

```bash
docker compose up
```

### 环境变量说明

| 变量                  | 必填 | 说明                                      |
| --------------------- | ---- | ----------------------------------------- |
| `OPENAI_API_KEY`      | ✅   | OpenAI API 密钥                           |
| `OPENAI_BASE_URL`     | ❌   | OpenAI 兼容 API 地址（默认官方地址）      |
| `MODEL`               | ❌   | 模型名称（默认 `gpt-4o`）                 |
| `WECHAT_APP_ID`       | ❌   | 微信公众号 AppID（发布草稿时需要）        |
| `WECHAT_APP_SECRET`   | ❌   | 微信公众号 AppSecret（发布草稿时需要）    |
| `WECHAT_ARTICLES_DIR` | ❌   | 往期文章目录（默认 `knowledge/articles/`） |

---

## 📁 项目结构

```
WechatOfficialAccountAssistant/
├── README.md
├── LICENSE
└── assistant_flow/
    ├── pyproject.toml                    # 项目配置与依赖
    ├── uv.lock                           # 锁定依赖版本
    ├── Dockerfile                        # Docker 镜像定义
    ├── .dockerignore                     # Docker 忽略文件
    ├── .env                              # 环境变量（需自行创建）
    ├── AGENTS.md                         # CrewAI 开发参考（AI 助手用）
    ├── knowledge/                        # 往期文章知识库
    │   └── articles/
    │       ├── ai-agent-revolution.md
    │       ├── content-creation-ai-tools.md
    │       └── prompt-engineering-guide.md
    ├── src/assistant_flow/
    │   ├── __init__.py
    │   ├── main.py                       # Flow 主流程编排
    │   ├── crews/                        # 智能体团队定义
    │   │   ├── content_crew/             # 内容创作团队
    │   │   │   ├── content_crew.py
    │   │   │   └── config/
    │   │   │       ├── agents.yaml       # Agent 角色定义
    │   │   │       └── tasks.yaml        # Task 任务定义
    │   │   └── publishing_crew/          # 发布团队
    │   │       ├── publishing_crew.py
    │   │       └── config/
    │   │           ├── agents.yaml
    │   │           └── tasks.yaml
    │   └── tools/                        # 自定义工具
    │       ├── __init__.py
    │       ├── custom_tool.py            # 自定义工具模板
    │       ├── langchain_tools.py        # 搜索/图片生成/格式校验
    │       └── llamaindex_tools.py       # 往期文章检索/风格分析/RAG
    └── output/                           # 生成的文章输出目录
        ├── wechat_article.md             # 最终文章
        └── wechat_article_meta.md        # 元信息
```

---

## 🛠️ 核心工具

### LlamaIndex 工具集（`llamaindex_tools.py`）

| 工具                   | 功能         | 说明                                     |
| ---------------------- | ------------ | ---------------------------------------- |
| `ArticleArchiveTool`   | 往期文章检索 | 基于向量索引的语义搜索，找到历史相关文章 |
| `StyleReferenceTool`   | 写作风格分析 | 自动分析语气、开头模式、金句频率等       |
| `InternalLinkingTool`  | 智能内部链接 | 发现相关往期文章，生成「相关阅读」板块   |
| `TopicGapAnalysisTool` | 主题空白分析 | 分析内容矩阵，发现选题差异化方向         |
| `KnowledgeRAGTool`     | 知识片段检索 | 从往期文章中提取可复用的数据、案例、金句 |

### LangChain 工具集（`langchain_tools.py`）

| 工具                        | 功能        | 说明                                |
| --------------------------- | ----------- | ----------------------------------- |
| `WebSearchTool`             | 网络搜索    | DuckDuckGo 搜索，获取最新资讯和数据 |
| `ImageGenerationTool`       | AI 图片生成 | DALL·E 生成封面图与插图            |
| `WeChatFormatValidatorTool` | 格式校验    | 检查标题长度、敏感词、排版结构等    |
| `PublishToDraftTool`        | 草稿发布    | 调用微信 API 推送到草稿箱           |

---

## 📝 往期文章知识库

在 `assistant_flow/knowledge/articles/` 目录下添加 `.md` 文件，即可自动被 LlamaIndex 索引，供风格分析、知识检索使用。

文章格式建议包含元信息：

```markdown
# 文章标题

> 发表日期: 2025-01-15 | 标签: AI, 自动化, 企业效率

正文内容...
```

支持的格式：`.md` 文件，自动提取标题、日期、标签等信息。

---

## 🔄 完整工作流

1. **接收标题** → 用户输入或 trigger payload 传入文章主题
2. **内容规划** → 分析往期文章 + 选题空白 + 设计大纲
3. **风格分析** → 学习往期写作风格，输出「风格指南卡」
4. **网络搜索** → 搜索最新资讯、数据、案例等素材
5. **文章撰写** → 结合所有素材撰写完整 Markdown 文章
6. **配图设计** → 设计封面图和内容插图方案
7. **排版优化** → 微信排版优化（标题层级、段落间距、留白等）
8. **审核校对** → 文字校对、逻辑检查、标注修改建议
9. **合规审查** → 检查敏感词、诱导用语、违规风险
10. **草稿发布** → 保存 Markdown 文件 + 推送草稿箱（可选）

---

## 📤 输出说明

执行完成后，会在 `assistant_flow/output/` 目录下生成：

- **`wechat_article.md`** — 完整的最终文章（Markdown 格式），可直接复制到微信后台
- **`wechat_article_meta.md`** — 文章元信息（标题、字数、时间等）

---

## ⚙️ 高级配置

### 切换 LLM 模型

编辑 `content_crew.py` 或 `publishing_crew.py` 中的 `_llm()` 方法：

```python
def _llm(self):
    return LLM(
        model="anthropic/claude-sonnet-4-20250514",  # 切换模型
        temperature=0.7,
    )
```

支持的 provider 格式：`openai/gpt-4o`、`anthropic/claude-sonnet-4`、`google/gemini-2.0-flash`、`ollama/llama3` 等。

### 自定义往期文章目录

设置环境变量指定自定义目录：

```bash
export WECHAT_ARTICLES_DIR=/path/to/your/articles
```

### 微信草稿发布（可选）

1. 在 `.env` 中配置 `WECHAT_APPID` 和 `WECHAT_APPSECRET`（需已认证的微信服务号）
2. 发布流程将自动调用 `PublishToDraftTool` 推送文章到微信草稿箱

---

## 📄 许可证

[MIT License](./LICENSE)

---

## 🙏 致谢

- [CrewAI](https://crewai.com) — 多智能体编排框架
- [LlamaIndex](https://www.llamaindex.ai) — 向量索引与 RAG 检索
- [LangChain](https://www.langchain.com) — LLM 应用开发框架

