from typing import Callable
import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

from agent.artifact import ArtifactStore

load_dotenv()


@dataclass
class ParamMeta:
    """
    Метаинформация о параметре приложения.

    Attributes:
        key (str): Ключ параметра.
        type (type): Тип значения.
        default (any): Значение по умолчанию.
        source (str): Источник значения.
        category (str): Категория параметра.
        description (str): Описание параметра.
        visible (bool): Видимость в интерфейсе.
        widget (str | None): Виджет для отображения.
        options (list | None): Опции для выбора.
        min_value (any | None): Минимальное значение.
        max_value (any | None): Максимальное значение.
        step (any | None): Шаг изменения.
        factory (callable | None): Фабрика для значения.
    """

    key: str
    type: type
    default: object
    source: str
    category: str
    description: str
    visible: bool = True
    widget: str | None = None
    options: list | None = None
    min_value: object | None = None
    max_value: object | None = None
    step: object | None = None
    factory: Callable | None = None


class AppParamsManager:
    """
    Менеджер параметров приложения с поддержкой Streamlit session_state.

    Attributes:
        params (dict[str, ParamMeta]): Описание всех параметров.
    """

    def __init__(self) -> None:
        self.params: dict[str, ParamMeta] = {}
        self._init_params()
        self._sync_with_env()
        self._sync_with_session_state()

    def _init_params(self) -> None:
        self.params = {
            "llm_model": ParamMeta(
                key="llm_model",
                type=str,
                default=os.getenv("LLM_MODEL_NAME", "gpt-4.1-nano"),
                source="env",
                category="llm",
                description="Название LLM модели",
                visible=True,
                widget="text_input",
            ),
            "llm_api_key": ParamMeta(
                key="llm_api_key",
                type=str,
                default=os.getenv("LLM_API_KEY", ""),
                source="sidebar",
                category="llm",
                description="API ключ LLM",
                visible=True,
                widget="text_input",
            ),
            "llm_temperature": ParamMeta(
                key="llm_temperature",
                type=float,
                default=float(os.getenv("LLM_TEMPERATURE", 0.7)),
                source="sidebar",
                category="llm",
                description="Температура генерации",
                visible=True,
                widget="slider",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
            ),
            "llm_max_iterations": ParamMeta(
                key="llm_max_iterations",
                type=int,
                default=int(os.getenv("LLM_MAX_ITERATIONS", 10)),
                source="sidebar",
                category="llm",
                description="Максимум итераций агента",
                visible=True,
                widget="slider",
                min_value=5,
                max_value=50,
                step=1,
            ),
            "llm_max_execution_time": ParamMeta(
                key="llm_max_execution_time",
                type=int,
                default=int(os.getenv("LLM_MAX_EXECUTION_TIME", 20)),
                source="sidebar",
                category="llm",
                description="Максимальное время выполнения агента (сек)",
                visible=True,
                widget="slider",
                min_value=10,
                max_value=200,
                step=1,
            ),
            "llm_enable_thinking": ParamMeta(
                key="llm_enable_thinking",
                type=bool,
                default=False,
                source="sidebar",
                category="llm",
                description="Включить режим 'thinking' (рассуждения)",
                visible=True,
                widget="selectbox",
                options=[False, True],
            ),
            "llm_base_url": ParamMeta(
                key="llm_base_url",
                type=str,
                default=os.getenv("LLM_MODEL_API_URL", ""),
                source="env",
                category="llm",
                description="Базовый URL для LLM API",
                visible=True,
                widget="text_input",
            ),
            "uploaded_data": ParamMeta(
                key="uploaded_data",
                type=object,
                default=None,
                source="internal",
                category="data",
                description="Загруженные данные.",
                visible=False,
            ),
            "max_dashboard_cols": ParamMeta(
                key="max_dashboard_cols",
                type=int,
                default=2,
                source="sidebar",
                category="dashboard",
                description="Максимум графиков в строке дашборда",
                visible=True,
                widget="slider",
                options=[1, 2, 3],
                min_value=1,
                max_value=3,
                step=1,
            ),
            "profiling_active": ParamMeta(
                key="profiling_active",
                type=bool,
                default=False,
                source="ui",
                category="profiling",
                description="Активен ли профайлинг",
                visible=False,
            ),
            "artifact_store": ParamMeta(
                key="artifact_store",
                type=object,
                default=None,
                source="internal",
                category="internal",
                description="Хранилище артефактов",
                visible=False,
                factory=ArtifactStore,
            ),
            "agent_service": ParamMeta(
                key="agent_service",
                type=object,
                default=None,
                source="internal",
                category="internal",
                description="Сервис агента",
                visible=False,
            ),
            "llm_use_history": ParamMeta(
                key="llm_use_history",
                type=str,
                default="Нет",
                source="sidebar",
                category="llm",
                description="Использовать историю сообщений",
                visible=True,
                widget="selectbox",
                options=["Нет", "Да"],
            ),
        }

    def _sync_with_env(self) -> None:
        for key, meta in self.params.items():
            if meta.source == "env" and key not in st.session_state:
                st.session_state[key] = meta.default

    def _sync_with_session_state(self) -> None:
        for key, meta in self.params.items():
            if key not in st.session_state:
                if key == "artifact_store":
                    st.session_state[key] = ArtifactStore()
                elif key == "agent_service":
                    st.session_state[key] = None
                elif key == "uploaded_data":
                    st.session_state[key] = None
                else:
                    st.session_state[key] = meta.default

    def get(self, key: str) -> object:
        """
        Получает значение параметра по ключу.

        Args:
            key (str): Ключ параметра.

        Returns:
            any: Значение параметра.
        """
        if key not in st.session_state:
            meta = self.params[key]
            if meta.factory is not None:
                st.session_state[key] = meta.factory()
            else:
                st.session_state[key] = meta.default
        return st.session_state[key]

    def set(self, key: str, value: object) -> None:
        """
        Устанавливает значение параметра по ключу.

        Args:
            key (str): Ключ параметра.
            value (any): Новое значение.
        """
        st.session_state[key] = value

    def reset(self) -> None:
        """
        Сбрасывает все параметры к значениям по умолчанию.
        """
        for key, meta in self.params.items():
            st.session_state[key] = meta.default

    def as_dict(self) -> dict[str, object]:
        """
        Возвращает все параметры в виде словаря.

        Returns:
            dict[str, any]: Словарь параметров.
        """
        return {key: self.get(key) for key in self.params}

    def render_sidebar(self) -> None:
        """
        Автоматически генерирует sidebar по описанию параметров.
        """
        categories = {}
        for key, meta in self.params.items():
            if meta.visible:
                categories.setdefault(meta.category, []).append(meta)
        for category, params in categories.items():
            with st.sidebar.expander(
                category.capitalize(), expanded=(category == "data")
            ):
                for meta in params:
                    if meta.widget == "slider":
                        st.slider(
                            meta.description,
                            min_value=meta.min_value,
                            max_value=meta.max_value,
                            value=self.get(meta.key),
                            step=meta.step,
                            key=meta.key,
                        )
                    elif meta.widget == "text_input":
                        st.text_input(
                            meta.description, value=self.get(meta.key), key=meta.key
                        )
                    elif meta.widget == "selectbox":
                        st.selectbox(
                            meta.description,
                            options=meta.options,
                            index=(
                                meta.options.index(self.get(meta.key))
                                if self.get(meta.key) in meta.options
                                else 0
                            ),
                            key=meta.key,
                        )
                    else:
                        self.set(meta.key, self.get(meta.key))


params_manager = AppParamsManager()
