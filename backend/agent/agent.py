from dataclasses import dataclass

import pandas as pd
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from backend.agent.agent_callback import ReasoningCallbackHandler, ToolCallbackHandler
from backend.artifacts.artifact import Artifact, artifact_factory
from backend.agent.pandas_agent import (
    create_pandas_dataframe_agent,
    extract_agent_output_text,
    normalize_agent_messages,
)
from backend.agent.prompts import agent_prompt
from backend.tools.impl import PandasTool, PlotlyTool, SQLTool, ValueTool
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from utils.params_manager import params_manager


@dataclass
class AgentResult:
    """
    Результат работы агента.

    Attributes:
        artifacts (list[Artifact]): Список артефактов, полученных в результате работы агента.
    """

    artifacts: list[Artifact]


class Agent:
    """
    Класс агента для анализа данных с помощью LLM и инструментов Pandas/Plotly.

    Args:
        df (pd.DataFrame): Исходный DataFrame для анализа.
        llm_base_url (str): URL для LLM API.
        llm_model (str): Название LLM модели.
        llm_api_key (str | None): API ключ для LLM.
        llm_temperature (float): Температура генерации.
        llm_max_iterations (int): Максимум итераций агента.
        llm_max_execution_time (int): Максимальное время выполнения агента (сек).
        llm_enable_thinking (bool): Включить режим рассуждения.

    Attributes:
        df (pd.DataFrame): Исходный DataFrame.
        llm_model (str): Название LLM модели.
        llm_api_key (str | None): API ключ.
        llm_temperature (float): Температура генерации.
        llm_max_iterations (int): Максимум итераций.
        llm_max_execution_time (int): Максимальное время выполнения.
        llm_enable_thinking (bool): Режим рассуждения.
        artifacts (list): Список артефактов.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str | None = None,
        llm_temperature: float = 0.7,
        llm_max_iterations: int = 8,
        llm_max_execution_time: int = 10,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
    ):
        self.df = df
        self._llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_temperature = llm_temperature
        self.llm_max_iterations = llm_max_iterations
        self.llm_max_execution_time = llm_max_execution_time
        self.llm_enable_thinking = llm_enable_thinking
        self.llm_chat_template_kwargs_enabled = llm_chat_template_kwargs_enabled
        self.db_runtime_config = db_runtime_config
        self.artifacts = []

        self.reasoning_callback = ReasoningCallbackHandler()
        self.tool_callback = ToolCallbackHandler()
        self.callback_manager = CallbackManager(
            [self.reasoning_callback, self.tool_callback]
        )

        llm_kwargs = {
            "model": llm_model,
            "base_url": llm_base_url,
            "api_key": llm_api_key,
            "streaming": False,
        }
        if llm_chat_template_kwargs_enabled:
            llm_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": llm_enable_thinking}
            }
        self.llm = ChatOpenAI(**llm_kwargs)
        self.agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df.copy(),
            tools=self._create_tools(),
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=llm_max_iterations,
            max_execution_time=llm_max_execution_time,
            prefix=agent_prompt,
        )

    def _create_tools(self) -> list[BaseTool]:
        """
        Создаёт инструменты для работы с данными.

        Returns:
            list[BaseTool]: Список инструментов для анализа данных.
        """
        tools: list[BaseTool] = [
            PlotlyTool(self.df),
            PandasTool(self.df),
            ValueTool(self.df),
        ]
        if self.db_runtime_config is not None:
            tools.append(
                SQLTool(
                    llm_base_url=self._llm_base_url,
                    llm_model=self.llm_model,
                    llm_api_key=self.llm_api_key,
                    llm_enable_thinking=self.llm_enable_thinking,
                    llm_chat_template_kwargs_enabled=self.llm_chat_template_kwargs_enabled,
                    db_runtime_config=self.db_runtime_config,
                    csv_loaded=False,
                    csv_session_id=None,
                    max_rows=200,
                )
            )
        return tools

    def _create_user_artifact(self, prompt: str) -> Artifact:
        """
        Создаёт артефакт пользователя на основе промпта.

        Args:
            prompt (str): Входной запрос пользователя.

        Returns:
            Artifact: Артефакт пользователя.
        """
        user_artifact = artifact_factory(type="user", data=prompt)
        params_manager.get("artifact_store").add(user_artifact)
        return user_artifact

    @staticmethod
    def _build_llm_history(artifacts: list[Artifact]) -> list[BaseMessage]:
        """
        Формирует список сообщений для LLM из артефактов.

        Args:
            artifacts (list[Artifact]): Список артефактов.

        Returns:
            list[BaseMessage]: Список сообщений для LLM.
        """
        return [
            a.history_message
            for a in artifacts
            if hasattr(a, "history_message") and a.history_message is not None
        ]

    def _build_prompt_for_llm(self, user_artifact: Artifact) -> list[BaseMessage]:
        """
        Формирует промпт для LLM с учётом истории сообщений.

        Args:
            user_artifact (Artifact): Артефакт пользователя.

        Returns:
            list[BaseMessage]: Список сообщений для LLM.
        """
        use_history = params_manager.get("llm_use_history") == "Да"
        artifact_store = params_manager.get("artifact_store")
        if use_history:
            return self._build_llm_history(artifact_store.get_chat_history())
        else:
            return self._build_llm_history([user_artifact])

    def _run_agent(self, prompt_for_llm: list[BaseMessage]) -> list[Artifact]:
        """
        Запускает агента для анализа данных.

        Args:
            prompt_for_llm (list[BaseMessage]): Список сообщений для LLM.

        Returns:
            list[Artifact]: Список артефактов, полученных в результате анализа.
        """
        self.reasoning_callback.artifacts.clear()
        self.tool_callback.artifacts.clear()
        self.tool_callback.tool_calls = 0
        try:
            runtime_config = {
                "callbacks": [self.reasoning_callback, self.tool_callback],
            }
            if self.llm_max_iterations:
                runtime_config["recursion_limit"] = max(4, self.llm_max_iterations * 2)
            agent_response = self.agent.invoke(
                {"messages": normalize_agent_messages(prompt_for_llm)},
                config=runtime_config,
            )
            artifacts = self.reasoning_callback.artifacts + self.tool_callback.artifacts
            if not artifacts:
                output_text = extract_agent_output_text(agent_response)
                artifacts = [
                    (
                        artifact_factory(type="text", data=output_text)
                        if output_text
                        else artifact_factory(
                            type="error",
                            data="❌ Агент не вернул ни одного артефакта.",
                        )
                    )
                ]
        except Exception as e:
            error_msg = f"❌ Ошибка при анализе: {str(e)}"
            artifacts = [artifact_factory(type="error", data=error_msg)]
        return artifacts

    def _add_artifacts_to_store(self, artifacts: list[Artifact]) -> None:
        """
        Добавляет артефакты в хранилище.

        Args:
            artifacts (list[Artifact]): Список артефактов для добавления.
        """
        for artifact in artifacts:
            params_manager.get("artifact_store").add(artifact)

    def analyze(self, prompt: str) -> list[Artifact]:
        """
        Анализирует данные на основе пользовательского запроса.

        Args:
            prompt (str): Запрос пользователя.

        Returns:
            list[Artifact]: Список артефактов, полученных в результате анализа.

        Example:
            >>> agent.analyze("Построй describe по данным")
        """
        user_artifact = self._create_user_artifact(prompt)
        prompt_for_llm = self._build_prompt_for_llm(user_artifact)
        artifacts = self._run_agent(prompt_for_llm)
        self._add_artifacts_to_store(artifacts)
        return artifacts

