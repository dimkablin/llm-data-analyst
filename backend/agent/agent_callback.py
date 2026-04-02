import re
from typing import Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage

from backend.artifacts.artifact import Artifact, artifact_factory


class ReasoningCallbackHandler(BaseCallbackHandler):
    """
    Обработчик колбэков для сбора рассуждений LLM и формирования артефактов.

    Attributes:
        artifacts (list[Artifact]): Список артефактов рассуждений.
    """

    def __init__(self) -> None:
        self.artifacts: list[Artifact] = []

    def on_llm_end(self, response: object, **kwargs) -> None:
        """
        Обрабатывает завершение работы LLM и добавляет артефакт рассуждения.

        Args:
            response (object): Ответ LLM.
            **kwargs: Дополнительные параметры.
        """
        if hasattr(response, "generations") and response.generations:
            text = response.generations[0][0].text
            filtered = re.sub(
                r"<think>[\s\S]*?<\/think>", "", text, flags=re.IGNORECASE
            ).strip()
            if filtered:
                self.artifacts.append(artifact_factory(type="text", data=filtered))


class ToolCallbackHandler(BaseCallbackHandler):
    """
    Обработчик колбэков для сбора артефактов от инструментов.

    Attributes:
        artifacts (list[Artifact]): Список артефактов.
        _handlers (dict[str, Callable[[object], None]]): Словарь обработчиков по типу результата.
    """

    def __init__(self) -> None:
        self.artifacts: list[Artifact] = []
        self.tool_calls: int = 0
        self._handlers: dict[str, Callable[[object], None]] = {
            "plot": self._handle_plot,
            "table": self._handle_table,
            "value": self._handle_value,
        }

    def on_tool_end(self, output: object, tool=None, **kwargs) -> None:
        """
        Обрабатывает завершение работы инструмента и добавляет артефакт.

        Args:
            output (object): Результат работы инструмента.
            tool (str, optional): Имя инструмента.
            **kwargs: Дополнительные параметры.
        """
        self.tool_calls += 1
        normalized_output = self._normalize_output(output)
        if normalized_output is None:
            return
        result_type = self._detect_type(normalized_output)
        handler = self._handlers.get(result_type, self._handle_value)
        handler(normalized_output)

    @staticmethod
    def _normalize_output(output: object) -> object | None:
        if isinstance(output, ToolMessage):
            artifact = getattr(output, "artifact", None)
            if isinstance(artifact, dict):
                return artifact
            return None
        return output

    def _detect_type(self, output: object) -> str:
        """
        Определяет тип результата инструмента.

        Args:
            output (object): Результат работы инструмента.

        Returns:
            str: Тип результата.
        """
        if isinstance(output, dict):
            if "plot" in output:
                return "plot"
            elif "table" in output:
                return "table"
        return "value"

    def _handle_plot(self, output: dict[str, object]) -> None:
        """
        Обрабатывает результат инструмента типа "plot".

        Args:
            output (dict[str, object]): Результат работы инструмента.
        """
        for name, fig in output["plot"].items():
            self.artifacts.append(artifact_factory(type="plot", data=fig, text=name))

    def _handle_table(self, output: dict[str, object]) -> None:
        """
        Обрабатывает результат инструмента типа "table".

        Args:
            output (dict[str, object]): Результат работы инструмента.
        """
        for name, table in output["table"].items():
            self.artifacts.append(artifact_factory(type="table", data=table, text=name))

    def _handle_value(self, output: object) -> None:
        """
        Обрабатывает результат инструмента типа "value".

        Args:
            output (object): Результат работы инструмента.
        """
        return


