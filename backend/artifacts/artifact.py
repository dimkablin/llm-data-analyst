import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

import markdown2
import pandas as pd
import plotly.graph_objects as go
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.data_access.dataframe_utils import numeric_summary_rows


@dataclass
class Artifact:
    """
    Базовый класс артефакта для хранения результатов анализа.

    Attributes:
        id (str): Уникальный идентификатор артефакта.
        data (object): Данные артефакта (текст, график, таблица и др.).
        text (str | None): Текстовое описание или имя артефакта.
        timestamp (datetime): Время создания артефакта.
        meta (dict[str, object]): Метаданные артефакта.
        normalized (object): Нормализованные данные для отображения.
        history_message (BaseMessage | None): Сообщение для истории LLM.
        type (str | None): Тип артефакта.
        role (str): Роль (ai или user).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: object = None
    text: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    meta: dict[str, object] = field(default_factory=dict)
    normalized: object = field(init=False, default=None)
    history_message: BaseMessage | None = field(init=False, default=None)
    type: str | None = None
    role: str = "ai"

    def __post_init__(self) -> None:
        """
        Инициализация артефакта: нормализация и генерация сообщения для истории.
        """
        self.normalized = self.normalize()
        self.history_message = self.generate_history_message()

    def normalize(self) -> object:
        """
        Возвращает нормализованные данные для отображения.

        Returns:
            object: Нормализованные данные.
        """
        return self.data

    def generate_history_message(self, *args, **kwargs) -> BaseMessage | None:
        """
        Генерирует сообщение для истории LLM.

        Returns:
            BaseMessage | None: Сообщение для истории.
        """
        content = str(self.text or self.data or "")
        if self.role == "user":
            return HumanMessage(content=content)
        else:
            return AIMessage(content=content)


@dataclass
class TextArtifact(Artifact):
    """
    Артефакт для хранения текстовых данных.
    """

    type: str = "text"

    def normalize(self) -> str:
        """
        Преобразует текст в HTML для отображения.

        Returns:
            str: HTML-строка.
        """
        text = self.data or ""
        html = markdown2.markdown(
            text,
            extras=[
                "fenced-code-blocks",
                "tables",
                "strike",
                "cuddled-lists",
                "break-on-newline",
                "code-friendly",
                "footnotes",
                "header-ids",
                "task_list",
                "smarty-pants",
                "target-blank-links",
            ],
        )
        html = re.sub(
            r"<h[1-6][^>]*>(.*?)</h[1-6]>",
            r'<div class="chat-title">\1</div>',
            html,
            flags=re.DOTALL,
        )
        return html

    def generate_history_message(self, *args, **kwargs) -> BaseMessage | None:
        """
        Генерирует сообщение для истории LLM на основе текста.

        Returns:
            BaseMessage | None: Сообщение для истории.
        """
        text = self.data
        if self.role == "user":
            return HumanMessage(content=text or "")
        else:
            return AIMessage(content=text or "")


@dataclass
class UserArtifact(TextArtifact):
    """
    Артефакт пользователя (текстовый, роль user).
    """

    type: str = "user"
    role: str = "user"


@dataclass
class TableArtifact(Artifact):
    type: str = "table"

    def normalize(self):  # pylint: disable=no-member
        table = self.data
        if isinstance(table, pd.Series):
            table = table.to_frame()
        for col in table.columns:
            if not pd.api.types.is_numeric_dtype(
                table[col]
            ) and not pd.api.types.is_bool_dtype(table[col]):
                table[col] = table[col].astype(str)
        return table

    def generate_history_message(  # pylint: disable=arguments-differ
        self, max_table_rows: int = 5, **kwargs
    ) -> BaseMessage | None:
        table_md = self.normalized.head(max_table_rows).to_markdown()
        summary_rows = numeric_summary_rows(self.normalized)
        summary_block = ""
        if summary_rows:
            summary_block = "\n\nnumeric_summary_rows_appended:\n" + pd.DataFrame(summary_rows).to_markdown(index=False)
        content = f"Первые {max_table_rows} строчек таблицы '{self.text}':\n{table_md}"
        content = content + summary_block
        return AIMessage(content=content)


@dataclass
class PlotArtifact(Artifact):
    """
    Артефакт для хранения графиков Plotly.
    """

    type: str = "plot"

    def normalize(self) -> go.Figure:  # pylint: disable=no-member
        """
        Нормализует график для отображения.

        Returns:
            go.Figure: Объект графика Plotly.
        """
        fig = self.data
        fig.update_layout(
            template="plotly_white",
            font=dict(size=13),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def generate_history_message(self, **kwargs) -> BaseMessage | None:  # pylint: disable=arguments-differ
        """
        Генерирует описание графика для истории LLM.

        Returns:
            BaseMessage | None: Сообщение для истории.
        """
        layout = self.normalized.layout
        title = getattr(layout, "title", None)
        title_text = getattr(title, "text", "") if title else ""
        xaxis = getattr(layout, "xaxis", None)
        xaxis_title = getattr(xaxis, "title", None)
        xaxis_text = getattr(xaxis_title, "text", "") if xaxis_title else ""
        yaxis = getattr(layout, "yaxis", None)
        yaxis_title = getattr(yaxis, "title", None)
        yaxis_text = getattr(yaxis_title, "text", "") if yaxis_title else ""
        trace_types = ", ".join(
            set(getattr(trace, "type", "-") for trace in self.normalized.data)
        )
        colors = ", ".join(
            str(getattr(getattr(trace, "marker", None), "color", "-"))
            for trace in self.normalized.data
            if hasattr(trace, "marker")
        )
        content = f"""Описание графика '{self.text}':
Заголовок: {title_text}
Тип(ы): {trace_types}
Цвета: {colors}
Ось X: {xaxis_text}
Ось Y: {yaxis_text}"""
        return AIMessage(content=content)


@dataclass
class ErrorArtifact(TextArtifact):
    type: ClassVar[str] = "error"


_artifact_type_map = {
    "text": TextArtifact,
    "user": UserArtifact,
    "table": TableArtifact,
    "plot": PlotArtifact,
    "error": ErrorArtifact,
}


def artifact_factory(
    artifact_type: str,
    data: object = None,
    text: str | None = None,
    meta: dict[str, object] | None = None,
) -> Artifact:
    """
    Фабрика для создания артефактов разных типов.

    Args:
        artifact_type (str): Тип артефакта (text, user, table, plot, error).
        data (object): Данные артефакта.
        text (str | None): Имя или описание.
        meta (dict[str, object] | None): Метаданные.

    Returns:
        Artifact: Экземпляр соответствующего класса артефакта.
    """
    meta = meta or {}
    cls = _artifact_type_map.get(artifact_type)
    if cls is None:
        raise ValueError(f"Неизвестный тип артефакта: {artifact_type}")
    return cls(data=data, text=text, meta=meta)


class ArtifactStore:
    """
    Хранилище артефактов для чата и дашборда.

    Attributes:
        artifacts (list[Artifact]): Все артефакты.
        dashboard_items (list[str]): ID артефактов на дашборде.
    """

    def __init__(self) -> None:
        self.artifacts: list[Artifact] = []
        self.dashboard_items: list[str] = []

    def add(self, artifact: Artifact) -> None:
        """
        Добавляет артефакт в хранилище.

        Args:
            artifact (Artifact): Артефакт для добавления.
        """
        self.artifacts.append(artifact)

    def get_chat_history(self) -> list[Artifact]:
        """
        Возвращает историю чата (все артефакты).

        Returns:
            list[Artifact]: Список артефактов.
        """
        return [a for a in self.artifacts]

    def get_dashboard_items(self) -> list[Artifact]:
        """
        Возвращает артефакты, добавленные на дашборд.

        Returns:
            list[Artifact]: Список артефактов на дашборде.
        """
        id_to_artifact = {a.id: a for a in self.artifacts}
        return [
            id_to_artifact[aid] for aid in self.dashboard_items if aid in id_to_artifact
        ]

    def mark_for_dashboard(self, artifact_id: str) -> None:
        """
        Помечает артефакт для отображения на дашборде.

        Args:
            artifact_id (str): ID артефакта.
        """
        if artifact_id not in self.dashboard_items:
            self.dashboard_items.append(artifact_id)
        # также помечаем артефакт для обратной совместимости
        for a in self.artifacts:
            if a.id == artifact_id:
                a.meta["dashboard"] = True

    def unmark_for_dashboard(self, artifact_id: str) -> None:
        """
        Убирает артефакт с дашборда.

        Args:
            artifact_id (str): ID артефакта.
        """
        if artifact_id in self.dashboard_items:
            self.dashboard_items.remove(artifact_id)
        for a in self.artifacts:
            if a.id == artifact_id and a.meta.get("dashboard", False):
                a.meta["dashboard"] = False

    def clear_chat(self) -> None:
        """
        Очищает историю чата.
        """
        self.artifacts = []

    def clear_dashboard(self) -> None:
        """
        Очищает дашборд.
        """
        self.dashboard_items.clear()
        for a in self.artifacts:
            if a.meta.get("dashboard", False):
                a.meta["dashboard"] = False
