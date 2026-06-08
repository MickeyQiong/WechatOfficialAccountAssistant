"""
LlamaIndex-based tools for WeChat Official Account article publishing.

Features:
  1. ArticleArchiveTool      — 索引 & 检索往期文章，支持语义搜索
  2. StyleReferenceTool      — 分析往期文章写作风格（语气、结构、金句模式）
  3. InternalLinkingTool     — 自动发现与新文章主题相关的往期文章，生成"相关阅读"链接
  4. TopicGapAnalysisTool    — 分析往期文章的主题分布，发现内容空白
  5. KnowledgeRAGTool        — 从往期文章中检索相关知识片段，丰富新文章内容

All tools share a global LlamaIndex index that is lazily built from
knowledge/articles/ directory on first use.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Index — 全局索引（懒加载，单例模式）
# ---------------------------------------------------------------------------

_global_index: Any = None
_global_documents: list[dict[str, str]] = []


def _get_articles_dir() -> Path:
    """确定往期文章目录路径。"""
    # 优先使用环境变量指定的目录
    env_dir = os.getenv("WECHAT_ARTICLES_DIR")
    if env_dir:
        return Path(env_dir)
    # 默认路径：assistant_flow/knowledge/articles
    # __file__ = .../assistant_flow/src/assistant_flow/tools/llamaindex_tools.py
    # .parent.parent.parent.parent = .../assistant_flow/
    return Path(__file__).resolve().parent.parent.parent.parent / "knowledge" / "articles"


def _load_articles() -> list[dict[str, str]]:
    """从目录中加载往期文章（返回标题、内容、文件名 等元信息）。"""
    articles_dir = _get_articles_dir()
    articles: list[dict[str, str]] = []

    if not articles_dir.exists():
        logger.warning(f"往期文章目录不存在: {articles_dir}")
        return articles

    for filepath in articles_dir.glob("*.md"):
        try:
            content = filepath.read_text(encoding="utf-8")
            # 从 Markdown 中提取标题（第一个 # 行）
            title = ""
            date = ""
            tags = ""
            for line in content.splitlines():
                if line.startswith("# ") and not title:
                    title = line[2:].strip()
                if line.startswith("> 发表日期:"):
                    match = re.search(r"发表日期:\s*(\S+)", line)
                    if match:
                        date = match.group(1)
                if line.startswith("> 发表日期:") and "标签:" in line:
                    match = re.search(r"标签:\s*(.+)", line)
                    if match:
                        tags = match.group(1)

            articles.append({
                "title": title or filepath.stem,
                "content": content,
                "filename": filepath.name,
                "date": date,
                "tags": tags,
                "word_count": str(len(content)),
            })
        except Exception as e:
            logger.warning(f"读取文章失败 {filepath}: {e}")

    return articles


def _build_index(force_rebuild: bool = False) -> Any:
    """构建 LlamaIndex 向量索引（懒加载，单例模式）。"""
    global _global_index, _global_documents

    if _global_index is not None and not force_rebuild:
        return _global_index

    articles = _load_articles()
    _global_documents = articles

    if not articles:
        logger.warning("往期文章为空，跳过索引构建")
        return None

    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.embeddings.openai import OpenAIEmbedding

        # 配置 embedding 模型
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # 构建文档
        llama_docs = []
        for article in articles:
            # 将元信息作为文档文本的一部分，增强语义检索
            doc_text = (
                f"标题: {article['title']}\n"
                f"日期: {article.get('date', '未知')}\n"
                f"标签: {article.get('tags', '')}\n"
                f"内容:\n{article['content']}"
            )
            llama_docs.append(
                Document(
                    text=doc_text,
                    metadata={
                        "title": article["title"],
                        "date": article.get("date", ""),
                        "tags": article.get("tags", ""),
                        "filename": article["filename"],
                    },
                )
            )

        _global_index = VectorStoreIndex.from_documents(llama_docs)
        logger.info(f"LlamaIndex 索引构建完成，共 {len(llama_docs)} 篇文章")

    except ImportError:
        logger.warning("llama-index 未安装，将使用关键词匹配降级方案")
        _global_index = None

    return _global_index


def _fallback_search(query: str) -> list[dict[str, str]]:
    """降级方案：基于关键词匹配的简单搜索。"""
    global _global_documents
    if not _global_documents:
        _global_documents = _load_articles()

    results = []
    query_lower = query.lower()
    for article in _global_documents:
        content_lower = article["content"].lower()
        title_lower = article["title"].lower()
        # 简单关键词匹配打分
        score = 0
        for word in query_lower.split():
            if word in title_lower:
                score += 3
            if word in content_lower:
                score += 1
        if score > 0:
            results.append((score, article))

    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results[:5]]


# ---------------------------------------------------------------------------
# Tool 1: ArticleArchiveTool — 往期文章检索
# ---------------------------------------------------------------------------

class ArticleArchiveInput(BaseModel):
    """输入：检索往期文章"""
    query: str = Field(
        ...,
        description="搜索查询，用于在往期文章中检索相关内容。可以是主题关键词或自然语言描述。",
    )
    top_k: int = Field(
        default=5,
        description="返回的文章数量（默认 5 篇）。",
    )


class ArticleArchiveTool(BaseTool):
    """检索往期文章 — 基于 LlamaIndex 向量索引的语义搜索。"""

    name: str = "article_archive_search"
    description: str = (
        "在往期微信公众号文章中检索与给定查询相关的内容。"
        "支持语义搜索，可根据主题、关键词、概念找到相似的历史文章。"
        "返回文章标题、摘要、日期和相关性片段。"
        "使用场景：查找与当前主题相关的往期文章、了解已覆盖过哪些话题。"
    )
    args_schema: Type[BaseModel] = ArticleArchiveInput

    def _run(self, query: str, top_k: int = 5) -> str:
        """执行语义检索。"""
        index = _build_index()

        if index is None:
            # 降级到关键词匹配
            articles = _fallback_search(query)[:top_k]
        else:
            try:
                retriever = index.as_retriever(similarity_top_k=top_k)
                nodes = retriever.retrieve(query)
                articles = [
                    {
                        "title": n.metadata.get("title", "未知"),
                        "date": n.metadata.get("date", ""),
                        "tags": n.metadata.get("tags", ""),
                        "content": n.get_text(),
                        "score": f"{n.score:.3f}",
                    }
                    for n in nodes
                ]
            except Exception as e:
                logger.warning(f"LlamaIndex 检索失败，降级到关键词匹配: {e}")
                articles = _fallback_search(query)[:top_k]

        if not articles:
            return "未找到与查询相关的往期文章。可能是往期文章库为空，请先在 knowledge/articles/ 目录下添加文章。"

        # 格式化输出
        parts = [f'## 📚 往期文章检索结果（查询: "{query}"）\n']
        for i, article in enumerate(articles, 1):
            title = article.get("title", "未知标题")
            date = article.get("date", "")
            tags = article.get("tags", "")
            score = article.get("score", "")
            content_preview = article.get("content", "")[:200].replace("\n", " ")

            parts.append(f"### {i}. {title}")
            if date:
                parts.append(f"📅 日期: {date}")
            if tags:
                parts.append(f"🏷️ 标签: {tags}")
            if score:
                parts.append(f"📊 相关度: {score}")
            parts.append(f"📝 摘要: {content_preview}...")
            parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 2: StyleReferenceTool — 写作风格分析
# ---------------------------------------------------------------------------

class StyleReferenceInput(BaseModel):
    """输入：分析往期文章风格"""
    topic: str = Field(
        ...,
        description="当前文章的主题或标题，用于在往期文章中寻找风格最接近的参考文章。",
    )


class StyleReferenceTool(BaseTool):
    """分析往期文章写作风格 — 为新文章提供风格参考。"""

    name: str = "style_reference"
    description: str = (
        "分析往期微信公众号文章的写作风格，为新文章提供风格参考。"
        "包括：语气特点（口语化/专业/轻松/严肃）、段落结构、开头方式、"
        "金句频率、互动引导方式、典型句式等。"
        "使用场景：需要保持公众号文风一致性，或参考成功文章的风格。"
    )
    args_schema: Type[BaseModel] = StyleReferenceInput

    def _run(self, topic: str) -> str:
        """分析往期文章风格。"""
        index = _build_index()

        if index is None:
            articles = _fallback_search(topic)[:3]
        else:
            try:
                retriever = index.as_retriever(similarity_top_k=3)
                nodes = retriever.retrieve(f"写作风格 语气 排版 {topic}")
                articles = [
                    {
                        "title": n.metadata.get("title", "未知"),
                        "content": n.get_text(),
                    }
                    for n in nodes
                ]
            except Exception:
                articles = _fallback_search(topic)[:3]

        if not articles:
            return "未找到往期文章，无法分析风格。请先在 knowledge/articles/ 目录下添加文章。"

        # 提取风格特征
        style_analysis = []
        style_analysis.append("## 🎨 往期文章风格分析\n")

        for i, article in enumerate(articles, 1):
            title = article.get("title", "未知")
            content = article.get("content", "")

            # 自动风格检测
            features = self._extract_style_features(content)

            style_analysis.append(f"### 参考文章 {i}：《{title}》\n")
            style_analysis.append(f"- **语气特点**: {features['tone']}")
            style_analysis.append(f"- **开头方式**: {features['opening_style']}")
            style_analysis.append(f"- **段落结构**: {features['paragraph_style']}")
            style_analysis.append(f"- **金句数量**: 发现 {features['quote_count']} 处亮点表达")
            style_analysis.append(f"- **互动引导**: {features['interaction']}")
            style_analysis.append(f"- **排版特点**: 短段落风格，频繁使用空行分隔")
            if features.get("typical_patterns"):
                style_analysis.append(f"- **典型句式**: {', '.join(features['typical_patterns'][:3])}")
            style_analysis.append("")

        # 总结统一风格
        style_analysis.append("### 📋 公众号统一风格总结\n")
        style_analysis.append("- **整体语调**: 亲切自然、像朋友聊天、有温度不冰冷")
        style_analysis.append("- **开头模式**: 用场景/问题/数据钩子抓注意力，前100字定生死")
        style_analysis.append("- **金句打法**: 每篇至少1-2句可截图分享的金句，用 > 💡 标记")
        style_analysis.append('- **结尾模板**: 总结 + 互动引导（"你怎么看？欢迎留言"）+ 往期推荐')
        style_analysis.append("- **排版铁律**: 每段≤4行、多换行、H2分段、加粗标重点")
        style_analysis.append("- **字数范围**: 1200-2000字，适合碎片化阅读")
        style_analysis.append("")
        style_analysis.append("⚠️ **写作建议**: 新文章请保持以上风格一致性，让读者感受到统一的品牌调性。")

        return "\n".join(style_analysis)

    def _extract_style_features(self, content: str) -> dict:
        """从文章内容中提取风格特征。"""
        features = {
            "tone": "中性",
            "opening_style": "直接切入",
            "paragraph_style": "标准段落",
            "quote_count": 0,
            "interaction": "无明确引导",
            "typical_patterns": [],
        }

        # 检测语气
        if any(w in content for w in ["你", "我们", "大家", "朋友"]):
            features["tone"] = "亲切口语化"
        if any(w in content for w in ["数据显示", "研究表明", "据统计"]):
            features["tone"] += " + 数据驱动"

        # 检测开头方式
        first_lines = "\n".join(content.splitlines()[:10])
        if "?" in first_lines:
            features["opening_style"] = "问题开头（引发思考）"
        elif any(w in first_lines for w in ["看到", "发现", "注意到"]):
            features["opening_style"] = "观察/场景开头"
        elif any(w in first_lines for w in ["数据", "统计", "%"]):
            features["opening_style"] = "数据/事实开头"

        # 检测金句
        quote_patterns = ["> 💡", "金句", "**", "——"]
        for p in quote_patterns:
            features["quote_count"] += content.count(p)

        # 检测互动引导
        if "欢迎留言" in content or "你怎么看" in content or "评论区" in content:
            features["interaction"] = "评论互动引导 ✓"
        if "转发" in content or "分享" in content:
            features["interaction"] += " + 分享引导 ✓"

        # 检测典型句式
        patterns = []
        if "你有没有" in content:
            patterns.append("「你有没有...」句式（共鸣引导）")
        if "简单说" in content or "打个比方" in content:
            patterns.append("比喻解释句式")
        if "条建议" in content or "个方法" in content or "个技巧" in content:
            patterns.append("清单式建议句式")
        features["typical_patterns"] = patterns

        return features


# ---------------------------------------------------------------------------
# Tool 3: InternalLinkingTool — 智能内部链接
# ---------------------------------------------------------------------------

class InternalLinkingInput(BaseModel):
    """输入：发现相关往期文章"""
    topic: str = Field(
        ...,
        description="当前文章的主题或标题，用于寻找最相关的往期文章作为内部链接。",
    )
    max_links: int = Field(
        default=3,
        description="建议的内部链接数量（默认 3 个）。",
    )


class InternalLinkingTool(BaseTool):
    """智能内部链接推荐 — 自动发现并生成「相关阅读」链接。"""

    name: str = "internal_linking"
    description: str = (
        "根据当前文章主题，在往期文章中自动发现最相关的内容，"
        "生成「相关阅读」内部链接推荐。帮助提升读者留存率和文章间流量互通。"
        "返回推荐的文章标题、简要说明和链接建议。"
        "使用场景：在新文章末尾自动添加「相关阅读」板块。"
    )
    args_schema: Type[BaseModel] = InternalLinkingInput

    def _run(self, topic: str, max_links: int = 3) -> str:
        """发现并推荐内部链接。"""
        index = _build_index()

        if index is None:
            articles = _fallback_search(topic)[:max_links]
        else:
            try:
                retriever = index.as_retriever(similarity_top_k=max_links)
                nodes = retriever.retrieve(topic)
                articles = [
                    {
                        "title": n.metadata.get("title", "未知"),
                        "date": n.metadata.get("date", ""),
                        "tags": n.metadata.get("tags", ""),
                        "score": f"{n.score:.3f}",
                    }
                    for n in nodes
                ]
            except Exception:
                articles = _fallback_search(topic)[:max_links]

        if not articles:
            return "未找到可关联的往期文章。建议手动添加相关阅读链接。"

        parts = ["## 🔗 智能内部链接推荐\n"]
        parts.append("以下往期文章与当前主题高度相关，建议在文末添加「相关阅读」板块：\n")

        for i, article in enumerate(articles, 1):
            title = article.get("title", "未知")
            date = article.get("date", "")
            tags = article.get("tags", "")
            score = article.get("score", "")

            parts.append(f"**{i}. 《{title}》**")
            if date:
                parts.append(f"   📅 {date}")
            if tags:
                parts.append(f"   🏷️ {tags}")
            if score:
                parts.append(f"   📊 相关度: {score}")
            parts.append(f"   🔗 建议链接文本: `[《{title}》](往期文章链接)`")
            parts.append("")

        # 生成可直接使用的 Markdown 相关阅读板块
        parts.append("### 📋 可直接复制到文末的「相关阅读」板块：\n")
        parts.append("```markdown")
        parts.append("---")
        parts.append("**📖 相关阅读：**")
        for article in articles:
            title = article.get("title", "")
            parts.append(f"- [《{title}》](往期文章链接)")
        parts.append("```")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 4: TopicGapAnalysisTool — 主题空白分析
# ---------------------------------------------------------------------------

class TopicGapInput(BaseModel):
    """输入：分析主题空白"""
    current_topic: str = Field(
        ...,
        description="当前准备撰写的文章主题，用于与往期文章对比，发现内容空白。",
    )


class TopicGapAnalysisTool(BaseTool):
    """主题空白分析 — 发现往期未覆盖但值得写的主题方向。"""

    name: str = "topic_gap_analysis"
    description: str = (
        "分析往期文章的主题分布，与当前拟写主题对比，"
        "发现内容空白、过度覆盖区域和新角度建议。"
        "帮助内容创作者保持主题多样性和新鲜感。"
        "使用场景：选题规划、判断是否需要从新角度切入。"
    )
    args_schema: Type[BaseModel] = TopicGapInput

    def _run(self, current_topic: str) -> str:
        """分析主题空白。"""
        articles = _load_articles()

        if not articles:
            return "往期文章库为空，无法进行主题空白分析。"

        # 收集所有标签
        all_tags: dict[str, int] = {}
        all_titles: list[str] = []
        for article in articles:
            all_titles.append(article.get("title", ""))
            tags_str = article.get("tags", "")
            if tags_str:
                for tag in tags_str.split(","):
                    tag = tag.strip()
                    if tag:
                        all_tags[tag] = all_tags.get(tag, 0) + 1

        parts = ["## 🔍 主题空白分析\n"]
        parts.append(f"**当前选题**: {current_topic}")
        parts.append(f"**往期文章总数**: {len(articles)} 篇\n")

        # 已有主题分布
        if all_tags:
            parts.append("### 往期主题标签分布\n")
            sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
            for tag, count in sorted_tags:
                bar = "█" * count
                parts.append(f"- {tag}: {bar} ({count}篇)")
            parts.append("")

        # 往期文章列表
        parts.append("### 往期文章目录\n")
        for i, title in enumerate(all_titles, 1):
            parts.append(f"{i}. {title}")
        parts.append("")

        # 空白分析建议
        parts.append("### 💡 内容策略建议\n")

        # 检查是否与往期过于相似
        similar_found = False
        for title in all_titles:
            if any(kw in title or kw in current_topic
                   for kw in current_topic.split() if len(kw) >= 2):
                similar_found = True
                break

        if similar_found:
            parts.append("- ⚠️ **主题重复风险**: 往期中已有与该主题相似的文章，建议从新角度切入或做更深度的解读")
        else:
            parts.append("- ✅ **新主题方向**: 该主题在往期中未覆盖，是内容矩阵的良好补充")

        # 基于标签分布的建议
        if all_tags:
            max_tag = max(all_tags, key=all_tags.get)
            parts.append(f"- 📊 **热点标签**: 「{max_tag}」出现最多（{all_tags[max_tag]}次），读者对此类内容兴趣度高")
            parts.append("- 💡 **互补方向建议**: 可考虑补充以下尚未覆盖的话题类型：案例研究、工具推荐、趋势预测、实操教程")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 5: KnowledgeRAGTool — 往期知识片段检索（RAG）
# ---------------------------------------------------------------------------

class KnowledgeRAGInput(BaseModel):
    """输入：从往期文章中检索知识片段"""
    question: str = Field(
        ...,
        description="需要从往期文章中寻找的知识点或问题描述。",
    )
    top_k: int = Field(
        default=3,
        description="返回的知识片段数量（默认 3 条）。",
    )


class KnowledgeRAGTool(BaseTool):
    """知识片段检索 — 从往期文章中提取可复用的知识点、数据、案例。"""

    name: str = "knowledge_rag"
    description: str = (
        "从往期微信公众号文章中检索与当前话题相关的知识片段。"
        "可以获取：往期用过的数据统计、案例分析、专业术语解释、"
        "引用来源等。帮助新文章在内容上更扎实、更有深度。"
        "使用场景：为当前文章补充事实论据、引用历史数据、复用成熟案例。"
    )
    args_schema: Type[BaseModel] = KnowledgeRAGInput

    def _run(self, question: str, top_k: int = 3) -> str:
        """从往期文章中检索知识片段。"""
        index = _build_index()

        if index is None:
            articles = _fallback_search(question)[:top_k]
        else:
            try:
                retriever = index.as_retriever(similarity_top_k=top_k)
                nodes = retriever.retrieve(question)
                articles = [
                    {
                        "title": n.metadata.get("title", "未知"),
                        "content": n.get_text(),
                        "score": f"{n.score:.3f}",
                    }
                    for n in nodes
                ]
            except Exception:
                articles = _fallback_search(question)[:top_k]

        if not articles:
            return "未从往期文章中找到相关知识点。建议通过网络搜索补充。"

        parts = [f'## 🧠 往期知识检索结果（查询: "{question}"）\n']
        parts.append("以下是从往期文章中检索到的相关知识片段，可参考引用：\n")

        for i, article in enumerate(articles, 1):
            title = article.get("title", "未知")
            content = article.get("content", "")
            score = article.get("score", "")

            parts.append(f"### 📄 来源 {i}：《{title}》")
            if score:
                parts.append(f"📊 相关度: {score}\n")

            # 提取最相关的段落（去元信息，保留正文）
            cleaned = re.sub(r'^>.*$', '', content, flags=re.MULTILINE)
            cleaned = re.sub(r'^#.*$', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'^---.*$', '', cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()

            # 截取前500字符作为知识片段
            snippet = cleaned[:500]
            if len(cleaned) > 500:
                snippet += "..."

            parts.append(f"```\n{snippet}\n```")
            parts.append(f"\n💡 *可引用上述片段中的数据、观点或案例到当前文章中。*\n")

        return "\n".join(parts)
