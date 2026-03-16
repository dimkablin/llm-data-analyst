import streamlit as st

from agent.artifact import Artifact
from utils.params_manager import params_manager


def process_user_message(prompt: str) -> None:
    """
    Обрабатывает сообщение пользователя и запускает анализ данных агентом.

    Args:
        prompt (str): Вопрос или запрос пользователя.
    """
    agent = params_manager.get("agent")
    with st.spinner("🧠 Думаю..."):
        agent.analyze(prompt)


def render_user_message(artifact: Artifact) -> None:
    """
    Отображает сообщение пользователя в чате.

    Args:
        artifact (Artifact): Артефакт пользователя.
    """
    st.markdown(
        f'<div class="msg-card user llm-fadein">'
        '<div class="msg-avatar">🧑‍💻</div>'
        f'<div class="markdown-content">{artifact.data}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_agent_message(artifact: Artifact) -> None:
    """
    Отображает сообщение агента в чате.

    Args:
        artifact (Artifact): Артефакт агента.
    """
    st.markdown(
        f'<div class="msg-card ai llm-fadein">'
        '<div class="msg-avatar">🧠</div>'
        '<div class="msg-content">'
        f'<div class="chat-markdown">{artifact.normalized}</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_error(artifact: Artifact) -> None:
    """
    Отображает сообщение об ошибке в чате.

    Args:
        artifact (Artifact): Артефакт ошибки.
    """
    st.markdown(
        f'<div class="msg-card error llm-fadein">'
        '<div class="msg-avatar">🧠</div>'
        '<div class="msg-content">'
        f'<div class="chat-markdown">{artifact.normalized}</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_table(artifact: Artifact) -> None:
    """
    Отображает таблицу в чате и позволяет добавить её на дашборд.

    Args:
        artifact (Artifact): Артефакт таблицы.
    """
    st.dataframe(
        artifact.data,
        use_container_width=True,
        hide_index=False,
    )
    if st.button(
        f"➕ Добавить таблицу на дашборд",
        key=f"add_to_dashboard_{artifact.id}_{artifact.text}",
    ):
        params_manager.get("artifact_store").mark_for_dashboard(artifact.id)
        st.success("Таблица добавлена на дашборд!")
        st.rerun()


def render_plot(artifact: Artifact) -> None:
    """
    Отображает график в чате и позволяет добавить его на дашборд.

    Args:
        artifact (Artifact): Артефакт графика.
    """
    st.plotly_chart(
        artifact.data,
        use_container_width=True,
        key=f"chat_plot_{artifact.id}_{artifact.text}",
    )
    if st.button(
        "➕ Добавить график на дашборд",
        key=f"add_to_dashboard_{artifact.id}_{artifact.text}",
    ):
        params_manager.get("artifact_store").mark_for_dashboard(artifact.id)
        st.success("График добавлен на дашборд!")
        st.rerun()


def render_unknown(artifact: Artifact) -> None:
    """
    Отображает сообщение о неизвестном типе артефакта.

    Args:
        artifact (Artifact): Неизвестный артефакт.
    """
    st.markdown(
        f'<div class="msg-card unknown llm-fadein">'
        f'<div class="msg-content">Неизвестный тип артефакта: {artifact.type}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_chat() -> None:
    """
    Отображает историю чата, сообщения пользователя, агента, таблицы и графики.
    """
    artifact_store = params_manager.get("artifact_store")
    chat_history = artifact_store.get_chat_history()

    with st.container(height=500, border=True):
        for artifact in chat_history:
            if artifact.role == "user":
                render_user_message(artifact)
            elif artifact.type == "table":
                render_table(artifact)
            elif artifact.type == "plot":
                render_plot(artifact)
            elif artifact.type == "text":
                render_agent_message(artifact)
            elif artifact.type == "error":
                render_error(artifact)
            else:
                render_unknown(artifact)

    prompt = st.chat_input("Введите ваш вопрос...")
    if prompt:
        if params_manager.get("uploaded_data") is None:
            st.warning("Пожалуйста, сначала загрузите файл с данными.")
            return
        process_user_message(prompt)
        st.rerun()
