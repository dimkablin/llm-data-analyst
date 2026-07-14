from __future__ import annotations

import unittest

import pandas as pd
import plotly.express as px

from backend.tools.plotly_express_compat import wrap_plotly_express


class PlotlyExpressCompatTests(unittest.TestCase):
    def test_wrap_accepts_showlegend_on_bar(self) -> None:
        wrapped = wrap_plotly_express(px)
        df = pd.DataFrame({"category": ["A", "B"], "value": [1, 2]})
        fig = wrapped.bar(df, x="category", y="value", showlegend=False)
        self.assertIs(fig.layout.showlegend, False)
