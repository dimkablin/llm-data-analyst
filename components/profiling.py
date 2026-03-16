import pandas as pd
import streamlit as st
from streamlit_ydata_profiling import st_profile_report
from ydata_profiling import ProfileReport

from utils.params_manager import params_manager


@st.cache_resource(show_spinner="Генерируем data profiling отчёт...")
def get_profile_report(df: pd.DataFrame) -> ProfileReport:
    """
    Генерирует отчёт профайлинга данных с помощью ydata-profiling.

    Args:
        df (pd.DataFrame): DataFrame для анализа.

    Returns:
        ProfileReport: Отчёт профайлинга.
    """
    return ProfileReport(df, title="Data Profiling Report", explorative=True)


def render_profiling() -> None:
    """
    Отображает отчёт профайлинга для загруженных данных.
    """
    st.markdown(
        '<div class="card-title">🧬 Data Profiling</div>', unsafe_allow_html=True
    )

    data = params_manager.get("uploaded_data")
    if data is None:
        st.warning("Сначала загрузите данные через Sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if st.button("🔄 Перегенерировать отчёт", key="refresh_profiling"):
        get_profile_report.clear()
        st.success("Профайлинг будет пересоздан при следующем отображении.")
        st.rerun()

    profile = get_profile_report(data)
    st_profile_report(profile)
    st.markdown("</div>", unsafe_allow_html=True)
