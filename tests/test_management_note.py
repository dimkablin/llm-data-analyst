from backend.agent.services.message_builder import polish_management_note_markdown


def test_polish_management_note_adds_heading_and_bold() -> None:
    raw = (
        "УПРАВЛЕНЧЕСКАЯ ЗАПИСКА\n\n"
        "1. Цель анализа\n"
        "Оценить портфель.\n\n"
        "2. Основные выводы\n"
        "- Концентрация в NREH: 22 млн руб.\n"
    )
    polished = polish_management_note_markdown(raw)
    assert "## УПРАВЛЕНЧЕСКАЯ ЗАПИСКА" in polished
    assert "**1. Цель анализа**" in polished
    assert "**NREH**" in polished
    assert "**22 млн руб.**" in polished
