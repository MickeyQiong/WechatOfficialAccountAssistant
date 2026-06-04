from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from assistant_flow.tools.langchain_tools import WebSearchTool


@CrewBase
class ContentCrew:
    """内容创作团队 — 规划 + 搜索 + 撰写

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
    def article_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["article_writer"],  # type: ignore[index]
            llm=self._llm(),
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────

    @task
    def plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["plan_task"],  # type: ignore[index]
        )

    @task
    def search_task(self) -> Task:
        return Task(
            config=self.tasks_config["search_task"],  # type: ignore[index]
            context=[self.plan_task()],
        )

    @task
    def write_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_task"],  # type: ignore[index]
            context=[self.plan_task(), self.search_task()],
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
