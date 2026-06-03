#!/usr/bin/env python
"""
微信公众号文章发表助手 — 主流程编排

Flow 流程:
  [用户输入标题]
       │
       ▼
  Step 1: ContentCrew（内容创作团队）
    内容规划师 → 网络搜索员 → 文章撰写者
    产出: 完整文章正文
       │
       ▼
  Step 2: PublishingCrew（发布团队）
    插图设计师 → 排版设计师 → 审核校对员 → 合规审查员 → 草稿发布员
    产出: WeChat 格式化终稿 + 封面图方案
       │
       ▼
  Step 3: 保存到本地 output/ 目录
"""

from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field

from crewai.flow import Flow, listen, start

from assistant_flow.src.assistant_flow.crews.content_crew.content_crew import ContentCrew
from assistant_flow.src.assistant_flow.crews.publishing_crew.publishing_crew import PublishingCrew


# ---------------------------------------------------------------------------
# Flow State — 结构化状态管理
# ---------------------------------------------------------------------------

class WechatArticleState(BaseModel):
    """微信公众号文章 Flow 状态"""

    # 输入
    title: str = Field(default="", description="用户输入的文章标题/主题")

    # ContentCrew 产出
    article_draft: str = Field(
        default="", description="ContentCrew 产出的文章初稿"
    )
    content_crew_result: str = Field(default="", description="ContentCrew 完整输出")

    # PublishingCrew 产出
    final_article: str = Field(
        default="", description="PublishingCrew 产出的最终文章"
    )
    cover_image_description: str = Field(
        default="", description="封面图设计方案"
    )
    compliance_report: str = Field(
        default="", description="合规审查报告"
    )

    # 元数据
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="创建时间",
    )


# ---------------------------------------------------------------------------
# WechatArticleFlow — 主流程
# ---------------------------------------------------------------------------

class WechatArticleFlow(Flow[WechatArticleState]):
    """微信公众号文章发表全流程编排。"""

    @start()
    def receive_title(self, crewai_trigger_payload: dict = None):
        """
        接收文章标题，作为整个流程的起点。

        支持两种输入方式：
        1. 通过 trigger payload 传入（自动化触发）
        2. 使用默认标题（手动执行）
        """
        print("\n" + "=" * 60)
        print("🚀 微信公众号文章发表助手启动")
        print("=" * 60)

        if crewai_trigger_payload:
            self.state.title = crewai_trigger_payload.get("title", "")
            print(f"📌 收到触发标题: {self.state.title}")
        else:
            # 默认演示标题
            self.state.title = "AI Agent 如何改变企业工作方式"
            print(f"📌 使用默认演示标题: {self.state.title}")

        print(f"⏰ 开始时间: {self.state.created_at}")
        print("=" * 60 + "\n")

    # ── Step 1: 内容创作 ─────────────────────────────────────────

    @listen(receive_title)
    def create_content(self):
        """
        调用 ContentCrew 进行内容创作：
        规划师 → 搜索员 → 撰写者
        """
        print("\n" + "─" * 60)
        print("📝 Step 1/2: 内容创作团队开始工作...")
        print("   流程: 内容规划 → 网络搜索 → 文章撰写")
        print("─" * 60 + "\n")

        result = (
            ContentCrew()
            .crew()
            .kickoff(inputs={"title": self.state.title})
        )

        self.state.article_draft = result.raw
        self.state.content_crew_result = result.raw

        print("\n" + "─" * 60)
        print("✅ Step 1/2 完成: 文章初稿已生成")
        print(f"   字数: {len(result.raw)} 字符")
        print("─" * 60 + "\n")

    # ── Step 2: 发布处理 ─────────────────────────────────────────

    @listen(create_content)
    def publish_content(self):
        """
        调用 PublishingCrew 进行发布处理：
        插图 → 排版 → 审校 → 合规审查 → 发布
        """
        print("\n" + "─" * 60)
        print("🎨 Step 2/2: 发布团队开始工作...")
        print("   流程: 插图设计 → 微信排版 → 审核校对 → 合规审查 → 草稿发布")
        print("─" * 60 + "\n")

        result = (
            PublishingCrew()
            .crew()
            .kickoff(
                inputs={
                    "title": self.state.title,
                    "article_content": self.state.article_draft,
                }
            )
        )

        self.state.final_article = result.raw

        print("\n" + "─" * 60)
        print("✅ Step 2/2 完成: 最终文章已生成")
        print(f"   字数: {len(result.raw)} 字符")
        print("─" * 60 + "\n")

    # ── Step 3: 保存 & 汇总 ──────────────────────────────────────

    @listen(publish_content)
    def save_and_summarize(self):
        """
        保存最终文章并输出汇总报告。
        """
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # 保存文章
        article_path = output_dir / "wechat_article.md"
        with open(article_path, "w", encoding="utf-8") as f:
            f.write(self.state.final_article)

        # 保存元信息
        meta_path = output_dir / "wechat_article_meta.md"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"# 文章元信息\n\n")
            f.write(f"- **标题**: {self.state.title}\n")
            f.write(f"- **创建时间**: {self.state.created_at}\n")
            f.write(f"- **初稿字数**: {len(self.state.article_draft)} 字符\n")
            f.write(f"- **终稿字数**: {len(self.state.final_article)} 字符\n")

        # 输出汇总
        print("\n" + "=" * 60)
        print("🎉 微信公众号文章发表流程完成！")
        print("=" * 60)
        print(f"📄 文章标题: {self.state.title}")
        print(f"📝 初稿字数: {len(self.state.article_draft)} 字符")
        print(f"✨ 终稿字数: {len(self.state.final_article)} 字符")
        print(f"💾 保存位置: {article_path.absolute()}")
        print(f"📋 元信息:   {meta_path.absolute()}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry Points
# ---------------------------------------------------------------------------

def kickoff():
    """
    命令行入口 — 执行完整流程。

    用法:
        crewai run
    """
    flow = WechatArticleFlow()
    flow.kickoff()


def plot():
    """
    生成流程图。

    用法:
        crewai flow plot
    """
    flow = WechatArticleFlow()
    flow.plot("wechat_article_flow")


def run_with_trigger():
    """
    通过 JSON trigger payload 执行流程。

    用法:
        uv run run_with_trigger '{"title": "你的文章标题"}'
    """
    import json
    import sys

    if len(sys.argv) < 2:
        print("用法: uv run run_with_trigger '{\"title\": \"你的标题\"}'")
        sys.exit(1)

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        print("错误: 请提供有效的 JSON payload")
        sys.exit(1)

    flow = WechatArticleFlow()
    flow.kickoff({"crewai_trigger_payload": trigger_payload})


if __name__ == "__main__":
    kickoff()
