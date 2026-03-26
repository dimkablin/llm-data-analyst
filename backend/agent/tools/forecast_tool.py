# Backward-compatibility shim — use `from tools.impl.forecast_tool import ...` in new code.
from backend.tools.impl.forecast_tool import *  # noqa: F401, F403
from backend.tools.impl.forecast_tool import ForecastTool, ForecastToolHelper


