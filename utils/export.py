import copy
import os
from abc import ABC, abstractmethod

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from agent.artifact import Artifact, ArtifactStore
from jinja2 import Environment, FileSystemLoader, select_autoescape

pio.templates.default = "plotly"

PLOTLY_WIDTH_CHAT = 800
PLOTLY_HEIGHT_CHAT = 400
PLOTLY_WIDTH_DASHBOARD = 800
PLOTLY_HEIGHT_DASHBOARD = 400

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")


def get_jinja_env(template_dir: str | None = None) -> Environment:
    """
    Создаёт и возвращает Jinja2 Environment для шаблонов HTML-отчётов.

    Args:
        template_dir (str | None): Путь к директории шаблонов.

    Returns:
        Environment: Jinja2 окружение.
    """
    template_dir = template_dir or TEMPLATE_DIR
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )


def plotly_to_html(
    obj: go.Figure,
    artifact_id: str,
    name: str,
    width: int,
    height: int,
    is_dashboard: bool = False,
) -> str:
    """
    Преобразует Plotly-объект в HTML для отчёта.

    Args:
        obj (go.Figure): Объект Plotly.
        artifact_id (str): ID артефакта.
        name (str): Имя графика.
        width (int): Ширина.
        height (int): Высота.
        is_dashboard (bool): Для дашборда или нет.

    Returns:
        str: HTML-код графика.
    """

    plot_obj = copy.deepcopy(obj)

    margin_config = (
        dict(l=60, r=60, t=60, b=60) if is_dashboard else dict(l=50, r=50, t=50, b=50)
    )

    plot_obj.update_layout(
        width=width,
        height=height,
        autosize=False,
        margin=margin_config,
        showlegend=True,
        legend=(
            dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            if is_dashboard
            else None
        ),
    )

    return pio.to_html(
        plot_obj,
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
        },
        div_id=f"plotly-div-{artifact_id}-{name}",
    )


class BaseExporter(ABC):
    """
    Базовый класс экспортёра артефактов в HTML-отчёты.

    Attributes:
        artifact_store (ArtifactStore): Хранилище артефактов.
        template_dir (str): Директория шаблонов.
        env (Environment): Jinja2 окружение.
    """

    def __init__(
        self, artifact_store: ArtifactStore, template_dir: str | None = None
    ) -> None:
        self.artifact_store = artifact_store
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), "..", "templates"
        )
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    @abstractmethod
    def get_artifacts(self) -> list[Artifact]:
        """
        Возвращает список артефактов для экспорта.

        Returns:
            list[Artifact]: Список артефактов.
        """
        pass

    @abstractmethod
    def render_html(self) -> str:
        """
        Генерирует HTML-отчёт по артефактам.

        Returns:
            str: HTML-отчёт.
        """
        pass

    def _serialize_artifact(self, artifact: Artifact) -> dict[str, object]:
        """
        Сериализует артефакт для отчёта.

        Args:
            artifact (Artifact): Артефакт.

        Returns:
            dict[str, object]: Словарь с сериализованными данными.
        """
        is_dashboard = hasattr(self, "is_dashboard") and self.is_dashboard
        if artifact.type == "table":
            normalized = artifact.normalized.to_html(
                index=True,
                classes="table table-sm table-striped",
            )
        elif artifact.type == "plot":
            if is_dashboard:
                width = PLOTLY_WIDTH_DASHBOARD
                height = PLOTLY_HEIGHT_DASHBOARD
            else:
                width = PLOTLY_WIDTH_CHAT
                height = PLOTLY_HEIGHT_CHAT
            normalized = plotly_to_html(
                artifact.normalized,
                artifact.id,
                artifact.text,
                width,
                height,
                is_dashboard,
            )
        else:
            normalized = artifact.normalized
        text_html = None
        if artifact.type == "text" or artifact.type == "user":
            text_html = artifact.normalized

        return {
            "id": artifact.id,
            "type": artifact.type,
            "normalized": normalized,
            "text": artifact.text,
            "text_html": text_html,
            "role": artifact.role,
            "timestamp": artifact.timestamp,
            "meta": artifact.meta,
        }


class ChatExporter(BaseExporter):
    """
    Экспортёр чата в HTML-отчёт.
    """

    is_dashboard = False

    def get_artifacts(self) -> list[Artifact]:
        return self.artifact_store.get_chat_history()

    def render_html(self) -> str:
        template = self.env.get_template("chat_report.html.j2")
        artifacts = [self._serialize_artifact(a) for a in self.get_artifacts()]
        return template.render(artifacts=artifacts)


class DashboardExporter(BaseExporter):
    """
    Экспортёр дашборда в HTML-отчёт.
    """

    is_dashboard = True

    def __init__(
        self,
        artifact_store: ArtifactStore,
        max_cols: int = 2,
        template_dir: str | None = None,
    ) -> None:
        super().__init__(artifact_store, template_dir)
        self.max_cols = max_cols

    def get_artifacts(self) -> list[Artifact]:
        return self.artifact_store.get_dashboard_items()

    def render_html(self) -> str:
        template = self.env.get_template("dashboard_report.html.j2")
        artifacts = [self._serialize_artifact(a) for a in self.get_artifacts()]
        return template.render(artifacts=artifacts, max_cols=self.max_cols)


@st.cache_data(show_spinner="Генерируем HTML-отчёт дашборда...")
def generate_dashboard_html_report(
    artifacts_data: list[dict[str, object]], max_cols: int
) -> str:
    """
    Генерирует HTML-отчёт дашборда по сериализованным артефактам и числу колонок.

    Args:
        artifacts_data (list[dict[str, object]]): Список сериализованных артефактов.
        max_cols (int): Максимальное число колонок.

    Returns:
        str: HTML-отчёт дашборда.
    """
    env = get_jinja_env()
    template = env.get_template("dashboard_report.html.j2")
    return template.render(artifacts=artifacts_data, max_cols=max_cols)


@st.cache_data(show_spinner="Генерируем HTML-отчёт чата...")
def generate_chat_html_report(artifacts_data: list[dict[str, object]]) -> str:
    """
    Генерирует HTML-отчёт чата по сериализованным артефактам.

    Args:
        artifacts_data (list[dict[str, object]]): Список сериализованных артефактов.

    Returns:
        str: HTML-отчёт чата.
    """
    env = get_jinja_env()
    template = env.get_template("chat_report.html.j2")
    return template.render(artifacts=artifacts_data)
