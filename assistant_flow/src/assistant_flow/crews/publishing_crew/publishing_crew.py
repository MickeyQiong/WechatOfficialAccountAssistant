from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from assistant_flow.tools.langchain_tools import (
    ImageGenerationTool,
    WeChatFormatValidatorTool,
)


@CrewBase
class PublishingCrew:
    """发布团队 — 插图 + 排版 + 审校 + 合规 + 发布

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
            temperature=0.5,
        )

    # ── Agents ────────────────────────────────────────────────────

    @agent
    def illustration_designer(self) -> Agent:
        return Agent(
            config=self.agents_config["illustration_designer"],  # type: ignore[index]
            llm=self._llm(),
            tools=[ImageGenerationTool()],
            verbose=True,
        )

    @agent
    def layout_designer(self) -> Agent:
        return Agent(
            config=self.agents_config["layout_designer"],  # type: ignore[index]
            llm=self._llm(),
            verbose=True,
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],  # type: ignore[index]
            llm=self._llm(),
            tools=[WeChatFormatValidatorTool()],
            verbose=True,
        )

    @agent
    def compliance_checker(self) -> Agent:
        return Agent(
            config=self.agents_config["compliance_checker"],  # type: ignore[index]
            llm=self._llm(),
            verbose=True,
        )

    @agent
    def draft_publisher(self) -> Agent:
        return Agent(
            config=self.agents_config["draft_publisher"],  # type: ignore[index]
            llm=self._llm(),
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────

    @task
    def illustrate_task(self) -> Task:
        return Task(
            config=self.tasks_config["illustrate_task"],  # type: ignore[index]
        )

    @task
    def layout_task(self) -> Task:
        return Task(
            config=self.tasks_config["layout_task"],  # type: ignore[index]
            context=[self.illustrate_task()],
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_task"],  # type: ignore[index]
            context=[self.layout_task()],
        )

    @task
    def compliance_task(self) -> Task:
        return Task(
            config=self.tasks_config["compliance_task"],  # type: ignore[index]
            context=[self.review_task()],
        )

    @task
    def publish_task(self) -> Task:
        return Task(
            config=self.tasks_config["publish_task"],  # type: ignore[index]
            context=[self.compliance_task()],
        )

    # ── Crew ──────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """创建发布团队。"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
