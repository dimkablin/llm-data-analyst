import pandas as pd

from backend.services import plot_recipes


def test_concentration_plot_specs_from_generic_segment_table() -> None:
    df = pd.DataFrame(
        {
            "channel": ["online", "partner", "retail"],
            "share_pct": [18.0, 63.0, 19.0],
        }
    )
    specs = plot_recipes.build_autogen_plot_specs_for_frames(
        [("channel_mix", df)],
    )

    assert specs
    assert all("risk" not in name for name, _, _ in specs)
