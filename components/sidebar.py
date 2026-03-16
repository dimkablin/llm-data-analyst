import pandas as pd
import streamlit as st

from components.chat import process_user_message
from utils.data_loader import load_csv_data, load_db_data
from utils.export import (
    ChatExporter,
    DashboardExporter,
    generate_chat_html_report,
    generate_dashboard_html_report,
)
from utils.params_manager import params_manager


def render_sidebar() -> None:
    """
    Отображает сайдбар с загрузкой данных, быстрыми промптами, настройками LLM и профайлингом.
    """
    if "pending_state_restore" in st.session_state:
        state_dict = st.session_state["pending_state_restore"]
        for key, value in state_dict.items():
            st.session_state[key] = value
        del st.session_state["pending_state_restore"]
        st.rerun()

    with st.sidebar.expander("📥 Загрузка данных", expanded=True):
        st.markdown(
            '<div class="sidebar-title">Загрузите CSV или подключитесь к базе данных</div>',
            unsafe_allow_html=True,
        )
        data_source = st.radio(
            "Источник данных:",
            ["CSV-файл", "База данных"],
            horizontal=True,
            key="data_source_radio",
        )
        data: pd.DataFrame | None = None
        error_msg = None
        if data_source == "CSV-файл":
            uploaded_file = st.file_uploader(
                "Выберите CSV-файл", type=["csv"], key="file_uploader"
            )
            if uploaded_file is not None:
                try:
                    data = load_csv_data(uploaded_file)
                    if data is not None:
                        st.success("Данные успешно загружены!")
                    else:
                        error_msg = "Ошибка загрузки данных из CSV."
                except Exception as e:
                    error_msg = f"Ошибка при чтении файла: {e}"
        else:
            st.markdown(
                "<div class='sidebar-section-label'>Параметры подключения к базе данных</div>",
                unsafe_allow_html=True,
            )
            with st.container():
                host = st.text_input("Host", value="localhost", key="db_host")
                port = st.text_input("Port", value="5432", key="db_port")
                user = st.text_input("User", value="", key="db_user")
                password = st.text_input("Password", type="password", key="db_password")
                database = st.text_input("Database", value="", key="db_name")
                table = st.text_input(
                    "Таблица (или SQL-запрос)", value="", key="db_table"
                )
                if st.button(
                    "🔗 Загрузить из БД", key="load_db_btn", use_container_width=True, help="НАХОДИТСЯ В РАЗРАБОТКЕ, ИСПОЛЬЗУЙТЕ НА СВОЙ СТРАХ И РИСК.",
                ):
                    data = load_db_data(host, port, user, password, database, table)
        if data is not None:
            params_manager.set("uploaded_data", data)
        if error_msg:
            st.error(error_msg)
        if params_manager.get("uploaded_data") is not None:
            if st.button(
                "🧹 Очистить загруженные данные",
                key="clear_uploaded_data",
                use_container_width=True,
            ):
                params_manager.set("uploaded_data", None)
                st.info("Данные удалены. Загрузите новые данные для работы.")

    with st.sidebar.expander("💬 Чат", expanded=False):
        st.markdown(
            "<div class='sidebar-quick-title'><b>Быстрые промпты:</b></div>",
            unsafe_allow_html=True,
        )
        QUICK_PROMPTS = [
            "Покажи основные статистики по данным",
            "Построй таблицу describe",
            "Построй красивый график по данным",
            "Расскажи про колонки в данных и что они значат",
            "Покажи главные инсайты в данных",
        ]

        for i, text in enumerate(QUICK_PROMPTS):
            btn = st.button(
                text,
                key=f"quick_prompt_{i}_sidebar",
                help="Отправить промпт в чат",
                use_container_width=True,
            )
            if btn:
                process_user_message(text)
                st.rerun()
        st.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
        if st.button(
            "🧹 Очистить чат", use_container_width=True, key="clear_chat_btn_sidebar"
        ):
            artifact_store = params_manager.get("artifact_store")
            if artifact_store:
                artifact_store.clear_chat()
            st.info("Чат очищен.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        artifact_store = params_manager.get("artifact_store")
        if artifact_store and len(artifact_store.get_chat_history()) > 0:
            exporter = ChatExporter(artifact_store)
            artifacts_data = [
                exporter._serialize_artifact(a)
                for a in artifact_store.get_chat_history()
            ]
            chat_html = generate_chat_html_report(artifacts_data)
            st.download_button(
                "📥 Экспорт чата (HTML)",
                chat_html,
                file_name="chat_report.html",
                mime="text/html",
                key="download_chat_html_btn",
            )

    with st.sidebar.expander("📊 Дашборд", expanded=False):
        st.slider(
            "Максимальное число артефактов в строке дашборда",
            min_value=params_manager.params["max_dashboard_cols"].min_value,
            max_value=params_manager.params["max_dashboard_cols"].max_value,
            value=params_manager.get("max_dashboard_cols"),
            step=params_manager.params["max_dashboard_cols"].step,
            key="max_dashboard_cols",
        )
        if st.button(
            "🧹 Очистить дашборд",
            key="clear_dashboard_btn_sidebar",
            use_container_width=True,
        ):
            artifact_store = params_manager.get("artifact_store")
            if artifact_store:
                artifact_store.clear_dashboard()
            st.info("Дашборд очищен.")
            st.rerun()
        artifact_store = params_manager.get("artifact_store")
        dashboard_items = artifact_store.get_dashboard_items() if artifact_store else []
        if dashboard_items:
            exporter = DashboardExporter(
                artifact_store, max_cols=params_manager.get("max_dashboard_cols")
            )
            artifacts_data = [
                exporter._serialize_artifact(a)
                for a in artifact_store.get_dashboard_items()
            ]
            dashboard_html = generate_dashboard_html_report(
                artifacts_data, params_manager.get("max_dashboard_cols")
            )
            st.download_button(
                "📥 Экспорт дашборда (HTML)",
                dashboard_html,
                file_name="dashboard_report.html",
                mime="text/html",
                key="download_dashboard_html_btn",
            )

    with st.sidebar.expander("🧠 LLM Agent", expanded=False):
        default_model = params_manager.get("llm_model")
        st.text_input(
            "Модель LLM (только чтение, задаётся в .env)",
            value=default_model,
            key="llm_model",
            disabled=True,
        )
        st.slider(
            "Температура (креативность)",
            min_value=params_manager.params["llm_temperature"].min_value,
            max_value=params_manager.params["llm_temperature"].max_value,
            value=params_manager.get("llm_temperature"),
            step=params_manager.params["llm_temperature"].step,
            help="Значение температуры для LLM при рассуждениях.  \n- Низкие значения → более предсказуемые ответы.  \n- Высокие значения → более креативные.",
            key="llm_temperature",
        )
        st.slider(
            "Максимум итераций LLM Agent",
            min_value=params_manager.params["llm_max_iterations"].min_value,
            max_value=params_manager.params["llm_max_iterations"].max_value,
            value=params_manager.get("llm_max_iterations"),
            step=params_manager.params["llm_max_iterations"].step,
            help="Максимальное количество шагов рассуждения и действия.  \n- Низкие значения → быстрые ответы, не глубокие рассуждения больше шанс ошибок.  \n- Высокие значения → медленнее ответы, глубже рассуждения, меньше шансов на ошибки.",
            key="llm_max_iterations",
        )
        st.slider(
            "Максимальное время выполнения агента (сек)",
            min_value=params_manager.params["llm_max_execution_time"].min_value,
            max_value=params_manager.params["llm_max_execution_time"].max_value,
            value=params_manager.get("llm_max_execution_time"),
            step=params_manager.params["llm_max_execution_time"].step,
            help="Ограничение по времени на выполнение одного запроса к LLM агенту.",
            key="llm_max_execution_time",
        )
        llm_enable_thinking_radio = st.radio(
            "Включить режим рассуждения (thinking)",
            options=["Нет", "Да"],
            index=1 if params_manager.get("llm_enable_thinking") else 0,
            key="llm_enable_thinking_radio",
            help="Если включено, агент будет использовать режим рассуждения (thinking) в LLM (llm_enable_thinking).",
        )
        if (llm_enable_thinking_radio == "Да") != params_manager.get(
            "llm_enable_thinking"
        ):
            params_manager.set("llm_enable_thinking", llm_enable_thinking_radio == "Да")
            st.rerun()
        llm_history_mode = st.radio(
            "Использовать историю сообщений?",
            options=params_manager.params["llm_use_history"].options,
            index=params_manager.params["llm_use_history"].options.index(
                params_manager.get("llm_use_history")
            ),
            key="llm_history_mode_radio",
            help="НАХОДИТСЯ В РАЗРАБОТКЕ, ИСПОЛЬЗУЙТЕ НА СВОЙ СТРАХ И РИСК.  \nИспользование истории сообщений.  \n- Да → LLM будет видеть весь диалог.  \n- Нет → только последнее сообщение пользователя.",
        )
        if llm_history_mode != params_manager.get("llm_use_history"):
            params_manager.set("llm_use_history", llm_history_mode)
            st.rerun()

    with st.sidebar.expander("🧬 Профайлинг", expanded=False):
        btn_text = (
            "Построить профайлинг"
            if not params_manager.get("profiling_active")
            else "Удалить профайлинг"
        )
        st.button(
            btn_text,
            use_container_width=True,
            key="toggle_profiling_btn_sidebar",
            help="Выполнить автоматический анализ и визуализацию структуры данных.  \n- Позволяет быстро увидеть основные характеристики и распределения признаков.  \n- Профайлинг строится с помощью ydata-profiling.",
        )
        if st.session_state.get("toggle_profiling_btn_sidebar"):
            params_manager.set(
                "profiling_active", not params_manager.get("profiling_active")
            )
            st.rerun()
