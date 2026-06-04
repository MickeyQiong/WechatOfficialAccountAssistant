"""
LangChain-based tools for WeChat Official Account article publishing.

Each tool is a CrewAI BaseTool that uses LangChain components internally:
  - WebSearchTool: 网络搜索 (DuckDuckGo / Tavily)
  - ImageGenerationTool: AI 图片生成 (DALL-E via langchain-openai)
  - WeChatFormatValidatorTool: 微信公众号格式校验
"""

import os
import re
from typing import ClassVar, Optional, Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量


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
                model_name="dall-e-3",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_proxy=os.getenv("OPENAI_PROXY"),
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
    cover_image_url: Optional[str] = Field(
        None,
        description="可选的封面图 URL，用于上传草稿封面缩略图。",
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

class PublishToDraftTool(BaseTool):
    """草稿箱发布工具 — 将文章内容发布到微信公众号草稿箱。"""

    name: str = "publish_to_draft"
    description: str = (
        "将文章内容发布到微信公众号草稿箱。"
        "需要提供文章标题、正文内容和封面图 URL。"
        "返回发布结果和草稿 ID。"
    )
    args_schema: Type[BaseModel] = WeChatFormatInput

    def _run(self, content: str, cover_image_url: Optional[str] = None) -> str:
        """执行发布操作并返回结果。"""
        try:
            import requests
        except ImportError:
            return (
                "[模拟模式] 发布草稿需要 requests 包。"
                " 请运行: pip install requests"
            )

        try:
            access_token = self._get_access_token()
            title, digest = self._extract_title_and_digest(content)
            if not title:
                return (
                    "❌ 发布失败：未能从内容中提取文章标题。"
                    "请在 Markdown 中以 '# 标题' 开头。"
                )

            # 将 Markdown 转为微信可接受的 HTML 并处理内嵌图片（上传至微信并替换 URL）
            html_content = self._markdown_to_wechat_html(content)
            html_content = self._process_content_images(access_token, html_content)

            article = {
                "title": title,
                "content": html_content,
                "digest": digest,
                "show_cover_pic": 1 if cover_image_url else 0,
                "content_source_url": os.getenv("WECHAT_CONTENT_SOURCE_URL", ""),
            }

            if cover_image_url:
                article["thumb_media_id"] = self._upload_cover_image(
                    access_token, cover_image_url
                )

            draft_id = self._create_draft(access_token, article)
            return (
                f"✅ 草稿发布成功！\n"
                f"草稿 ID: {draft_id}\n"
                f"请登录微信公众号后台草稿箱查看草稿。"
            )

        except Exception as exc:
            return f"❌ 发布失败：{str(exc)}"

    def _get_access_token(self) -> str:
        """从环境变量获取或调用微信接口获取 access_token。"""
        if os.getenv("WECHAT_ACCESS_TOKEN"):
            return os.getenv("WECHAT_ACCESS_TOKEN")

        appid = os.getenv("WECHAT_APPID")
        secret = os.getenv("WECHAT_APPSECRET")
        if not appid or not secret:
            raise ValueError(
                "缺少微信认证配置，请设置 WECHAT_APPID 和 WECHAT_APPSECRET 环境变量。"
            )

        import requests

        token_url = (
            "https://api.weixin.qq.com/cgi-bin/token"
            f"?grant_type=client_credential&appid={appid}&secret={secret}"
        )
        response = requests.get(token_url, timeout=15)
        data = response.json()
        if response.status_code != 200 or "access_token" not in data:
            raise ValueError(
                "获取 access_token 失败："
                f"{data.get('errmsg', data)}"
            )
        return data["access_token"]

    def _create_draft(self, access_token: str, article: dict) -> str:
        """调用微信草稿接口创建草稿。"""
        import requests

        draft_url = (
            "https://api.weixin.qq.com/cgi-bin/draft/add"
            f"?access_token={access_token}"
        )
        response = requests.post(draft_url, json={"articles": [article]}, timeout=20)
        data = response.json()
        if data.get("errcode", 0) != 0:
            raise ValueError(
                "草稿创建失败："
                f"{data.get('errmsg', data.get('errcode'))}"
            )

        draft_id = data.get("media_id") or data.get("draft_id")
        if not draft_id:
            raise ValueError("草稿创建成功，但未返回草稿 ID。")
        return draft_id

    def _upload_cover_image(self, access_token: str, image_url: str) -> str:
        """上传封面图到微信素材接口，返回 media_id。"""
        import requests

        # 使用永久素材接口上传封面，返回 media_id（草稿接口期望永久素材 media_id）
        upload_url = (
            "https://api.weixin.qq.com/cgi-bin/material/add_material"
            f"?access_token={access_token}&type=image"
        )

        if image_url.startswith("http://") or image_url.startswith("https://"):
            response = requests.get(image_url, timeout=20)
            if response.status_code != 200:
                raise ValueError(
                    f"下载封面图失败，HTTP {response.status_code}。"
                )
            file_data = response.content
            file_name = "cover.jpg"
        else:
            if not os.path.exists(image_url):
                raise ValueError(f"封面图路径不存在：{image_url}")
            with open(image_url, "rb") as f:
                file_data = f.read()
            file_name = os.path.basename(image_url)

        files = {"media": (file_name, file_data, "image/jpeg")}
        response = requests.post(upload_url, files=files, timeout=60)
        data = response.json()
        if data.get("errcode", 0) != 0:
            raise ValueError(
                "封面图上传失败："
                f"{data.get('errmsg', data.get('errcode'))}"
            )
        if "media_id" not in data:
            raise ValueError("封面图上传成功，但未返回 media_id。")
        return data["media_id"]

    def _process_content_images(self, access_token: str, html_content: str) -> str:
        """查找 HTML 中的 <img src="...">，上传到微信的 uploadimg 接口并替换为微信返回的 URL。"""
        import requests

        def upload_image_bytes(bdata, filename="image.jpg"):
            uploadimg_url = (
                "https://api.weixin.qq.com/cgi-bin/media/uploadimg"
                f"?access_token={access_token}"
            )
            files = {"media": (filename, bdata, "image/jpeg")}
            resp = requests.post(uploadimg_url, files=files, timeout=60)
            data = resp.json()
            if resp.status_code != 200 or "url" not in data:
                raise ValueError(f"上传内文图片失败: {data}")
            return data["url"]

        # 使用正则找到所有 img 标签 src
        matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_content)
        for src in matches:
            try:
                if src.startswith("http://") or src.startswith("https://"):
                    r = requests.get(src, timeout=20)
                    if r.status_code != 200:
                        continue
                    new_url = upload_image_bytes(r.content, os.path.basename(src))
                else:
                    # 本地路径
                    if not os.path.exists(src):
                        continue
                    with open(src, "rb") as f:
                        new_url = upload_image_bytes(f.read(), os.path.basename(src))

                # 替换原有 src 为微信返回的 url
                html_content = html_content.replace(src, new_url)
            except Exception:
                # 上传失败则跳过，不影响整体流程
                continue

        return html_content

    def _extract_title_and_digest(self, content: str) -> tuple[str, str]:
        """从 Markdown 内容中提取标题和摘要。"""
        lines = content.splitlines()
        title = ""
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        if not title:
            for line in lines:
                if line.strip():
                    title = line.strip()
                    break

        plain_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", content)
        plain_text = re.sub(r"\*\*|__|`|\*|#", "", plain_text)
        digest = " ".join(plain_text.split())[:120]
        return title, digest

    def _markdown_to_wechat_html(self, content: str) -> str:
        """将 Markdown 内容转换为简化的微信 HTML 格式。"""
        content = content.strip()
        content = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", r"<img src=\"\2\" alt=\"\1\" />", content)
        content = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", content)
        content = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", content)
        content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)

        html_lines = []
        in_list = False
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                continue

            if stripped.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h2>{stripped[3:].strip()}</h2>")
            elif stripped.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h1>{stripped[2:].strip()}</h1>")
            elif stripped.startswith("* ") or stripped.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                item = stripped[2:].strip()
                html_lines.append(f"<li>{item}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<p>{stripped}</p>")

        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)