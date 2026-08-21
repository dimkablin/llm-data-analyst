from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    invalid_argument = "INVALID_ARGUMENT"
    unauthorized_source = "UNAUTHORIZED_SOURCE"
    source_not_found = "SOURCE_NOT_FOUND"
    source_unavailable = "SOURCE_UNAVAILABLE"
    column_not_found = "COLUMN_NOT_FOUND"
    column_type_mismatch = "COLUMN_TYPE_MISMATCH"
    filter_type_mismatch = "FILTER_TYPE_MISMATCH"
    query_rejected = "QUERY_REJECTED"
    query_timeout = "QUERY_TIMEOUT"
    series_empty = "SERIES_EMPTY"
    series_too_short = "SERIES_TOO_SHORT"
    duplicate_period = "DUPLICATE_PERIOD"
    missing_periods = "MISSING_PERIODS"
    irregular_frequency = "IRREGULAR_FREQUENCY"
    non_finite_value = "NON_FINITE_VALUE"
    future_target_leakage = "FUTURE_TARGET_LEAKAGE"
    future_covariates_incomplete = "FUTURE_COVARIATES_INCOMPLETE"
    model_capability_mismatch = "MODEL_CAPABILITY_MISMATCH"
    model_not_available = "MODEL_NOT_AVAILABLE"
    model_input_rejected = "MODEL_INPUT_REJECTED"
    resource_exhausted = "RESOURCE_EXHAUSTED"
    inference_timeout = "INFERENCE_TIMEOUT"
    internal_error = "INTERNAL_ERROR"


class ChronosMCPError(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        field: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.retryable = retryable
        self.details = details or {}
