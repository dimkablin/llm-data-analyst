import streamlit as st
from agent.artifact import Artifact, ArtifactStore

from utils.params_manager import params_manager


def render_dashboard_table(artifact: Artifact, artifact_store: ArtifactStore) -> None:
    """
    Отображает таблицу на дашборде и позволяет удалить её.

    Args:
        artifact (Artifact): Артефакт таблицы.
        artifact_store (ArtifactStore): Хранилище артефактов.
    """
    st.markdown(
        f'<div class="card-title">{artifact.text}</div>', unsafe_allow_html=True
    )
    st.dataframe(
        artifact.data,
        use_container_width=True,
        hide_index=False,
    )
    if st.button("Удалить", key=f"remove_dashboard_{artifact.id}"):
        artifact_store.unmark_for_dashboard(artifact.id)
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_dashboard_plot(artifact: Artifact, artifact_store: ArtifactStore) -> None:
    """
    Отображает график на дашборде и позволяет удалить его.

    Args:
        artifact (Artifact): Артефакт графика.
        artifact_store (ArtifactStore): Хранилище артефактов.
    """
    st.markdown(
        f'<div class="card-title">{artifact.text}</div>', unsafe_allow_html=True
    )
    st.plotly_chart(
        artifact.data,
        use_container_width=True,
        key=f"dashboard_plot_{artifact.id}_{artifact.text}",
    )
    if st.button("Удалить", key=f"remove_dashboard_{artifact.id}"):
        artifact_store.unmark_for_dashboard(artifact.id)
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_dashboard() -> None:
    """
    Отображает дашборд с таблицами и графиками.
    """
    artifact_store: ArtifactStore = params_manager.get("artifact_store")
    artefacts = artifact_store.get_dashboard_items()
    n_artefacts = len(artefacts)
    max_cols = params_manager.get("max_dashboard_cols")

    if n_artefacts == 0:
        st.info(
            "Нет добавленных графиков и таблиц. Добавьте график или таблицу через чат."
        )
        return

    with st.container(height=500, border=True):
        for start in range(0, n_artefacts, max_cols):
            end = min(start + max_cols, n_artefacts)
            row_artefacts = artefacts[start:end]
            cols = st.columns(len(row_artefacts), gap="large")
            for i, artefact in enumerate(row_artefacts):
                with cols[i]:
                    if artefact.type == "table":
                        render_dashboard_table(artefact, artifact_store)
                    elif artefact.type == "plot":
                        render_dashboard_plot(artefact, artifact_store)
