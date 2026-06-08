from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from assistant_flow.tools.langchain_tools import WebSearchTool
from assistant_flow.tools.llamaindex_tools import (
    ArticleArchiveTool,
    InternalLinkingTool,
    KnowledgeRAGTool,
    StyleReferenceTool,
    TopicGapAnalysisTool,
)


@CrewBase
class ContentCrew:
    """内容创作团队 — 规划 + 风格参考 + 搜索 + 撰写

    利用 LlamaIndex 分析往期文章风格、检索相关知识片段、
    自动生成内部链接推荐，确保新文章与公众号品牌调性一致。

    LLM 配置: 使用 CrewAI LLM（底层通过 litellm 调用 OpenAI API，
    与 langchain_openai.ChatOpenAI 使用相同的 API 端点）。
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ── LLM 配置（使用 crewai.LLM，底层 litellm 与 langchain_openai 同 API）──
    def _llm(self):
        return LLM(
            model="openai/gpt-4o",
            temperature=0.7,
        )

    # ── Agents ────────────────────────────────────────────────────

    @agent
    def content_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["content_planner"],  # type: ignore[index]
            llm=self._llm(),
            tools=[
                ArticleArchiveTool(),
                StyleReferenceTool(),
                TopicGapAnalysisTool(),
            ],
            verbose=True,
        )

    @agent
    def style_referencer(self) -> Agent:
        return Agent(
            config=self.agents_config["style_referencer"],  # type: ignore[index]
            llm=self._llm(),
            tools=[
                ArticleArchiveTool(),
                StyleReferenceTool(),
                KnowledgeRAGTool(),
            ],
            verbose=True,
        )

    @agent
    def web_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config["web_searcher"],  # type: ignore[index]
            llm=self._llm(),
            tools=[WebSearchTool()],
            verbose=True,
        )

    @agent
    def image_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["image_generator"],  # type: ignore[index]
            llm=self._llm(),
            verbose=True,
        )

    @agent
    def article_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["article_writer"],  # type: ignore[index]
            llm=self._llm(),
            tools=[
                InternalLinkingTool(),
                KnowledgeRAGTool(),
            ],
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────

    @task
    def plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["plan_task"],  # type: ignore[index]
        )

    @task
    def style_reference_task(self) -> Task:
        return Task(
            config=self.tasks_config["style_reference_task"],  # type: ignore[index]
            context=[self.plan_task()],
        )

    @task
    def search_task(self) -> Task:
        return Task(
            config=self.tasks_config["search_task"],  # type: ignore[index]
            context=[self.plan_task(), self.style_reference_task()],
        )

    @task
    def write_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_task"],  # type: ignore[index]
            context=[self.plan_task(), self.search_task(), self.style_reference_task()],
        )

    @task
    def generate_image_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_image_task"],  # type: ignore[index]
            context=[self.plan_task(), self.write_task()],
        )

    # ── Crew ──────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """创建内容创作团队。"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
