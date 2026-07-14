from __future__ import annotations

import pandas as pd

from backend.agent.services.finalization import fallback_text


def test_fallback_text_does_not_claim_safe_answer_without_artifacts() -> None:
    text = fallback_text("Покажи динамику продаж", pd.DataFrame({"sales": [1, 2]}))

    assert "Без подтвержденных артефактов" not in text
    assert "безопасный ответ" not in text.lower()
    assert "не удалось" in text.lower() or "не смог" in text.lower()
