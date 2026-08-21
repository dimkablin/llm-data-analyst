"""Cross-section charts should use generic table shape, not domain lookup rules."""

from __future__ import annotations

import pandas as pd

from backend.services.plot_recipes import _build_cross_section_comparison_specs


def test_cross_section_comparison_prefers_multi_row_metric_table() -> None:
    lookup = pd.DataFrame([{"entity": "A", "score": 0.1}])
    comparison = pd.DataFrame(
        [
            {"entity": "A", "score": 0.1, "amount": 10.0},
            {"entity": "B", "score": 0.4, "amount": 20.0},
            {"entity": "C", "score": -0.2, "amount": 15.0},
        ]
    )

    specs = _build_cross_section_comparison_specs(
        [
            ("entity_lookup", lookup),
            ("entity_comparison", comparison),
        ],
        intent="comparison",
        base_meta={"autogen": True},
    )

    assert specs
    _, _, meta = specs[0]
    assert meta.get("source_table") == "entity_comparison"
