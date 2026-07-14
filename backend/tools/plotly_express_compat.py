"""Plotly Express compatibility shims for the session sandbox.

Some Plotly Express versions reject layout kwargs (e.g. ``showlegend``) on
``px.bar`` / ``px.line`` even though models often pass them. We peel those
kwargs off and apply them via ``fig.update_layout()`` after the chart is built.
"""
from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

# kwargs accepted by fig.update_layout but not by every px.* in all versions
_PX_LAYOUT_KWARGS = frozenset({"showlegend"})

# common chart constructors — avoid wrapping colors, set_mapbox_style, etc.
_PX_CHART_FUNCS = frozenset(
    {
        "area",
        "bar",
        "bar_polar",
        "box",
        "choropleth",
        "choropleth_map",
        "density_contour",
        "density_heatmap",
        "ecdf",
        "funnel",
        "funnel_area",
        "histogram",
        "icicle",
        "line",
        "line_3d",
        "line_geo",
        "line_map",
        "line_polar",
        "parallel_categories",
        "parallel_coordinates",
        "pie",
        "scatter",
        "scatter_3d",
        "scatter_geo",
        "scatter_map",
        "scatter_matrix",
        "scatter_polar",
        "scatter_ternary",
        "strip",
        "sunburst",
        "timeline",
        "treemap",
        "violin",
    }
)


def _wrap_px_chart_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        layout_updates = {k: kwargs.pop(k) for k in list(kwargs) if k in _PX_LAYOUT_KWARGS}
        fig = fn(*args, **kwargs)
        if layout_updates:
            fig.update_layout(**layout_updates)
        return fig

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


def wrap_plotly_express(px_module: Any) -> Any:
    """Return a px-like module with layout-kwarg tolerant chart functions."""
    mod = types.ModuleType(getattr(px_module, "__name__", "plotly.express"))
    mod.__doc__ = getattr(px_module, "__doc__", None)
    for attr in dir(px_module):
        if attr.startswith("_"):
            continue
        val = getattr(px_module, attr)
        if attr in _PX_CHART_FUNCS and callable(val):
            setattr(mod, attr, _wrap_px_chart_fn(val))
        else:
            setattr(mod, attr, val)
    return mod
