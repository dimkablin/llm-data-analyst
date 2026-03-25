import streamlit as st
from dotenv import load_dotenv
from agent.agent import Agent
from components.chat import render_chat
from components.dashboard import render_dashboard
from components.profiling import render_profiling
from components.sidebar import render_sidebar
from utils.params_manager import params_manager

load_dotenv()

@st.cache_data
def load_css(file_path: str) -> str:
    """
    Загружает CSS из файла.

    Args:
        file_path (str): Путь к CSS-файлу.

    Returns:
        str: Содержимое CSS-файла.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


st.markdown(
    f'<style>{load_css("templates/styles.css")}</style>', unsafe_allow_html=True
)

st.markdown(
    """
<div class="header-block">
    <div class="header-flex">
        <span class="header-logo">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
                <rect x="8" y="24" width="3" height="8" rx="1.5" fill="white"/>
                <rect x="15" y="16" width="3" height="16" rx="1.5" fill="white"/>
                <rect x="22" y="12" width="3" height="20" rx="1.5" fill="white"/>
                <rect x="29" y="6" width="3" height="26" rx="1.5" fill="white"/>
            </svg>
        </span>
        <span class="header-title">LLM Data Analyst</span>
    </div>
    <div class="header-subtitle">Аналитик данных на базе LLM</div>
</div>
""",
    unsafe_allow_html=True,
)

def main() -> None:
    """
    Основная функция приложения. Запускает Streamlit-интерфейс и инициализирует агента.
    """
    st.set_page_config(
        page_title="LLM Data Analyst", layout="wide", initial_sidebar_state="expanded"
    )

    render_sidebar()

    input_df = params_manager.get("uploaded_data")
    agent_service = params_manager.get("agent_service")

    if input_df is not None:
        llm_model = params_manager.get("llm_model")
        llm_api_key = params_manager.get("llm_api_key")
        llm_base_url = params_manager.get("llm_base_url")
        llm_temperature = params_manager.get("llm_temperature")
        llm_max_iterations = params_manager.get("llm_max_iterations")
        llm_max_execution_time = params_manager.get("llm_max_execution_time")
        llm_enable_thinking = params_manager.get("llm_enable_thinking")

        if not llm_api_key:
            st.error(
                "❌ API ключ не найден! Установите LLM_API_KEY в .env файле или в настройках."
            )
            return

        if (
            agent_service is None
            or id(agent_service.df) != id(input_df)
            or getattr(agent_service.agent, "llm_model", None) != llm_model
            or getattr(agent_service.agent, "llm_temperature", None) != llm_temperature
            or getattr(agent_service.agent, "llm_max_iterations", None)
            != llm_max_iterations
            or getattr(agent_service.agent, "llm_max_execution_time", None)
            != llm_max_execution_time
            or getattr(agent_service.agent, "llm_enable_thinking", None)
            != llm_enable_thinking
        ):

            with st.spinner("🧠 Инициализация агента..."):
                params_manager.set(
                    "agent",
                    Agent(
                        df=input_df,
                        llm_base_url=llm_base_url,
                        llm_model=llm_model,
                        llm_api_key=llm_api_key,
                        llm_temperature=llm_temperature,
                        llm_max_iterations=llm_max_iterations,
                        llm_max_execution_time=llm_max_execution_time,
                        llm_enable_thinking=llm_enable_thinking,
                    ),
                )

    dashboard_col, chat_col = st.columns([2, 1], gap="small")

    with dashboard_col:
        st.markdown(
            "<div class='section-header'><span class='section-icon'>📊</span>DashBoard</div>",
            unsafe_allow_html=True,
        )
        render_dashboard()

    with chat_col:
        st.markdown(
            "<div class='section-header'><span class='section-icon'>🧠</span>LLM Chat</div>",
            unsafe_allow_html=True,
        )
        render_chat()

    if params_manager.get("profiling_active"):
        st.markdown("<hr class='hr-large-margin'>", unsafe_allow_html=True)
        render_profiling()


if __name__ == "__main__":
    main()
