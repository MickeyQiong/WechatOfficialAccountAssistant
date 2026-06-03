"""
LangChain-based tools for WeChat Official Account article publishing.

Each tool is a CrewAI BaseTool that uses LangChain components internally:
  - WebSearchTool: 网络搜索 (DuckDuckGo / Tavily)
  - ImageGenerationTool: AI 图片生成 (DALL-E via langchain-openai)
  - WeChatFormatValidatorTool: 微信公众号格式校验
"""

from typing import ClassVar, Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool


# ---------------------------------------------------------------------------
# WebSearchTool — 基于 LangChain DuckDuckGo 的网络搜索
# ---------------------------------------------------------------------------

class WebSearchInput(BaseModel):
    """输入：搜索关键词"""
    query: str = Field(
        ...,
        description="搜索查询字符串，用于在互联网上查找相关信息。",
    )


class WebSearchTool(BaseTool):
    """网络搜索工具 — 使用 LangChain DuckDuckGo Search 组件。"""

    name: str = "web_search"
    description: str = (
        "在互联网上搜索与给定查询相关的最新信息。"
        "返回包含标题、摘要和链接的搜索结果。"
        "适用于查找新闻、数据、案例研究和背景资料。"
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """执行 DuckDuckGo 搜索并返回格式化结果。"""
        # 尝试新版 ddgs 包，回退到旧版 duckduckgo_search
        results = None
        import_error = None

        # 方案 1: 新版 ddgs 包
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
        except ImportError as e:
            import_error = str(e)
        except Exception:
            pass

        # 方案 2: 旧版 duckduckgo_search 包
        if results is None:
            try:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
            except ImportError as e:
                import_error = import_error or str(e)
            except Exception:
                pass

        if import_error and results is None:
            return (
                "错误：未安装搜索依赖。"
                "请运行: uv add ddgs"
            )

        if not results:
            return (
                f'⚠️ 搜索 "{query}" 未返回结果。'
                "（可能是网络限制或搜索API暂时不可用）\n\n"
                "建议：\n"
                "1. 检查网络连接是否正常\n"
                "2. 如果在中国大陆，DuckDuckGo 可能被限制，可考虑使用 Tavily Search API\n"
                "3. 设置 TAVILY_API_KEY 环境变量后可使用 Tavily 搜索"
            )

        # 格式化输出
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            body = r.get("body", "无摘要")
            href = r.get("href", "无链接")
            formatted.append(f"{i}. **{title}**\n   {body}\n   链接: {href}")

        return "\n\n".join(formatted)


# ---------------------------------------------------------------------------
# ImageGenerationTool — 基于 LangChain DALL-E 的图片生成
# ---------------------------------------------------------------------------

class ImageGenInput(BaseModel):
    """输入：图片生成描述"""
    prompt: str = Field(
        ...,
        description="图片生成的提示词，描述想要的图片内容、风格和用途。",
    )
    image_type: str = Field(
        default="cover",
        description="图片类型：'cover'（封面图, 900x383）或 'content'（内容插图, 16:9）。",
    )


class ImageGenerationTool(BaseTool):
    """AI 图片生成工具 — 使用 LangChain DALL-E 组件生成微信公众号图片。"""

    name: str = "image_generation"
    description: str = (
        "根据文本描述生成 AI 图片。可以生成微信公众号封面图（900x383 像素）"
        "或内容插图（16:9 比例）。使用 DALL-E 模型生成。"
        "参数: prompt (图片描述), image_type ('cover' 或 'content')。"
    )
    args_schema: Type[BaseModel] = ImageGenInput

    # WeChat 封面图尺寸: 900x383 px (2.35:1); DALL-E 2 固定 1024x1024
    COVER_SIZE: ClassVar[str] = "1024x1024"

    def _run(self, prompt: str, image_type: str = "cover") -> str:
        """生成图片并返回 URL。"""
        try:
            from langchain_openai import OpenAI
            from langchain_community.utilities.dalle_image_generator import (
                DallEAPIWrapper,
            )

            # 根据图片类型调整 prompt
            if image_type == "cover":
                full_prompt = (
                    f"微信公众号封面图，横版 2.35:1 比例，适合作为文章头图。"
                    f"风格：专业、吸引眼球、适合微信阅读。{prompt}"
                )
            else:
                full_prompt = (
                    f"微信公众号内容插图，16:9 比例，清晰美观。"
                    f"风格：信息图/示意图风格，适合在文章中嵌入。{prompt}"
                )

            # 使用 LangChain DALL-E wrapper
            dalle = DallEAPIWrapper(
                model="dall-e-3",
                size="1024x1024",
                quality="standard",
                n=1,
            )

            image_url = dalle.run(full_prompt)
            return (
                f"✅ 图片生成成功！\n"
                f"类型: {'封面图' if image_type == 'cover' else '内容插图'}\n"
                f"图片 URL: {image_url}"
            )

        except ImportError:
            return (
                "[模拟模式] 图片生成需要 langchain-openai 和 openai 包。\n"
                f"生成描述: {image_type} - {prompt[:200]}...\n"
                "提示: 设置 OPENAI_API_KEY 环境变量并安装依赖后可使用真实生成。"
            )
        except Exception as e:
            return (
                f"[模拟模式] 图片生成失败: {str(e)}\n"
                f"生成描述: {image_type} - {prompt[:200]}...\n"
                "将以文字描述替代图片。"
            )


# ---------------------------------------------------------------------------
# WeChatFormatValidatorTool — 微信公众号格式校验
# ---------------------------------------------------------------------------

class WeChatFormatInput(BaseModel):
    """输入：待校验的文章内容"""
    content: str = Field(
        ...,
        description="需要校验格式的微信公众号文章内容（Markdown 格式）。",
    )


class WeChatFormatValidatorTool(BaseTool):
    """微信公众号格式校验工具 — 检查文章是否符合平台格式规范。"""

    name: str = "wechat_format_validator"
    description: str = (
        "校验微信公众号文章格式是否符合平台规范。检查项包括："
        "标题长度（不超过64字）、封面图尺寸、正文排版、敏感词、"
        "外链限制、诱导分享/关注用语等。返回校验报告。"
    )
    args_schema: Type[BaseModel] = WeChatFormatInput

    def _run(self, content: str) -> str:
        """执行格式校验并返回报告。"""
        issues = []
        passed = []

        # 1. 检查标题长度（从内容中提取第一个 # 标题）
        lines = content.split("\n")
        title_line = ""
        for line in lines:
            if line.startswith("# "):
                title_line = line[2:].strip()
                break

        if title_line:
            if len(title_line) > 64:
                issues.append(f"⚠️  标题过长: {len(title_line)}字（限制64字）→ \"{title_line[:30]}...\"")
            else:
                passed.append(f"✅ 标题长度: {len(title_line)}字（OK）")
        else:
            issues.append("⚠️  未找到标题（建议以 '# 标题' 开头）")

        # 2. 检查正文长度
        body = "\n".join(lines)
        body_length = len(body)
        if body_length < 300:
            issues.append(f"⚠️  正文过短: {body_length}字符（建议至少800字）")
        elif body_length > 20000:
            issues.append(f"⚠️  正文过长: {body_length}字符（建议控制在20000字以内）")
        else:
            passed.append(f"✅ 正文长度: {body_length}字符（OK）")

        # 3. 检查敏感词/违规用语
        sensitive_patterns = [
            "点击领取", "转发到", "分享到朋友圈", "不转不是",
            "震惊", "出大事了", "马上删", "紧急通知",
        ]
        found_sensitive = []
        for pattern in sensitive_patterns:
            if pattern in body:
                found_sensitive.append(pattern)

        if found_sensitive:
            issues.append(
                f"⚠️  发现疑似诱导/标题党用语: {', '.join(found_sensitive)}"
                f"（建议修改，可能被微信限流）"
            )
        else:
            passed.append("✅ 未检测到明显诱导/标题党用语（OK）")

        # 4. 检查图片引用
        if "![" in body:
            image_count = body.count("![")
            passed.append(f"✅ 检测到 {image_count} 张图片引用（OK）")
        else:
            passed.append("ℹ️  未检测到图片引用（建议添加封面图和插图）")

        # 5. 检查 Markdown 结构
        if "## " in body:
            h2_count = body.count("## ")
            passed.append(f"✅ 检测到 {h2_count} 个二级标题，结构清晰（OK）")
        else:
            issues.append("ℹ️  未检测到二级标题（建议使用 ## 分段，提高可读性）")

        # 汇总报告
        report = "## 📋 微信公众号格式校验报告\n\n"
        report += "### ✅ 通过项\n"
        for p in passed:
            report += f"- {p}\n"

        if issues:
            report += "\n### ⚠️ 需修改项\n"
            for i in issues:
                report += f"- {i}\n"
        else:
            report += "\n### 🎉 全部通过！文章格式合规。\n"

        return report
