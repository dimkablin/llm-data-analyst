from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


def make_json_safe(obj: Any) -> Any:
    """Convert common pandas/numpy/python objects into JSON-serializable values.

    This function is intentionally conservative:
    - timestamps become ISO strings;
    - missing values become None;
    - numpy scalars become native Python scalars;
    - NaN/Inf become None, because strict JSON cannot safely represent them;
    - dict/list/tuple/set are handled recursively.
    """

    if obj is None:
        return None

    # Containers first: do not call pd.isna() on dict/list/array,
    # because it may return an array instead of a single bool.
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list | tuple | set):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.datetime64):
            if obj.ndim == 0:
                return make_json_safe(obj[()])
            return [make_json_safe(value) for value in obj]
        return make_json_safe(obj.tolist())

    if isinstance(obj, pd.Series):
        return make_json_safe(obj.tolist())

    if isinstance(obj, pd.DataFrame):
        return make_json_safe(obj.to_dict(orient="split"))

    if isinstance(obj, np.dtype | pd.api.extensions.ExtensionDtype):
        return str(obj)

    # Missing pandas/numpy values.
    # Covers: pd.NA, pd.NaT, np.nan, np.datetime64("NaT") in most cases.
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # Datetime-like values.
    if isinstance(obj, pd.Period):
        return str(obj)


    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, np.datetime64):
        try:
            if np.isnat(obj):
                return None
        except Exception:
            pass
        return pd.Timestamp(obj).isoformat()

    if isinstance(obj, datetime | date | time):
        return obj.isoformat()

    # Duration-like values.
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()

    if isinstance(obj, np.timedelta64):
        try:
            if np.isnat(obj):
                return None
        except Exception:
            pass
        return pd.Timedelta(obj).isoformat()

    if isinstance(obj, timedelta):
        return obj.total_seconds()

    # Numpy/native scalar values.
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, Decimal):
        value = float(obj)
        return value if math.isfinite(value) else None

    return obj


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas scalar and array types."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-renamed
        safe_obj = make_json_safe(obj)
        if safe_obj is not obj:
            return safe_obj
        return super().default(obj)
