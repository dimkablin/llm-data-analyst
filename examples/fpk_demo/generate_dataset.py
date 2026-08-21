from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

SEED = 20260729
HISTORY_START = date(2023, 1, 1)
HISTORY_END = date(2025, 12, 31)
STATS_START = date(2025, 1, 1)
STATS_END = date(2025, 12, 31)


@dataclass(frozen=True)
class Branch:
    csi_name: str
    stats_name: str
    isoo_column: str
    weight: float
    average_distance: float
    demand_bias: float
    quality_bias: float
    incident: bool = False


BRANCHES = (
    Branch("С-ЗАП", "СЕВ.ЗАП.", "с_зап", 1.12, 760, 0.01, 0.8),
    Branch("МОСК", "МОСК.", "моск", 1.35, 650, 0.02, 1.2),
    Branch("ГОРЬК", "ГОРЬК.", "горьк", 0.75, 720, -0.01, 0.4),
    Branch("С-КАВ", "С-КАВ.", "с_кав", 1.00, 980, 0.01, 0.1),
    Branch("ПРИВ", "ПРИВ.", "прив", 0.82, 870, -0.01, 0.3),
    Branch("КБШ", "КУЙБ.", "кбш", 0.78, 840, 0.00, 0.4),
    Branch("УР", "СВЕРД.", "ур", 0.86, 1_050, 0.00, 0.1),
    Branch("З-СИБ", "З-СИБ.", "з_сиб", 0.72, 1_200, 0.01, -0.6, True),
    Branch("В-СИБ", "В-СИБ.", "в_сиб", 0.62, 1_450, 0.00, -0.7, True),
    Branch("ДВОСТ", "ДВОСТ.", "двост", 0.60, 1_700, 0.01, -0.8, True),
)

STATS_COLUMNS = (
    "pass_turnover",
    "pass_count",
    "car_turnover",
    "seat_turnover",
    "structure",
    "date",
    "sys_section",
    "metric",
    "add_time",
    "updated_at",
    "is_deleted",
)

CSI_COMPONENTS = (
    "безопасность",
    "дорожный_набор",
    "поездка_с_детьми",
    "покупка_билетов",
    "ирс_попутчик",
    "постельные_принадлежности",
    "предоплаченное_питание",
    "ржд_бонус",
    "работа_проводников",
    "санитарное_состояние",
    "стоимость_поездки",
    "техническое_состояние",
    "уровень_комфорта",
    "услуги_вагона_ресторана",
)

CSI_COLUMNS = (
    "type",
    *CSI_COMPONENTS,
    "date",
    "индекс_удовлетворенности_пас",
    "add_time",
    "structure",
    "индекс_потребительской_лояльност",
    "sys_section",
)

CSI_PLAN_BASE = {
    "безопасность": 93.0,
    "дорожный_набор": 88.0,
    "поездка_с_детьми": 86.0,
    "покупка_билетов": 92.0,
    "ирс_попутчик": 87.0,
    "постельные_принадлежности": 90.0,
    "предоплаченное_питание": 86.0,
    "ржд_бонус": 89.0,
    "работа_проводников": 91.0,
    "санитарное_состояние": 89.0,
    "стоимость_поездки": 84.0,
    "техническое_состояние": 88.0,
    "уровень_комфорта": 87.0,
    "услуги_вагона_ресторана": 83.0,
}

INCIDENT_PENALTY = {
    "безопасность": 0.2,
    "дорожный_набор": 1.5,
    "поездка_с_детьми": 2.5,
    "покупка_билетов": 0.8,
    "ирс_попутчик": 1.0,
    "постельные_принадлежности": 3.5,
    "предоплаченное_питание": 3.5,
    "ржд_бонус": 0.5,
    "работа_проводников": 4.5,
    "санитарное_состояние": 6.5,
    "стоимость_поездки": 2.0,
    "техническое_состояние": 3.0,
    "уровень_комфорта": 5.5,
    "услуги_вагона_ресторана": 4.5,
}

ISOO_BRANCH_COLUMNS = (
    "с_зап",
    "моск",
    "горьк",
    "с_кав",
    "прив",
    "кбш",
    "ур",
    "з_сиб",
    "в_сиб",
    "двост",
)

ISOO_CHANNEL_COLUMNS = (
    "еисц",
    "почта_оао_ржд",
    "портал_генерального_директора_оао",
    "почта_ао_фпк",
    "почта_генерального_директора_ао_ф",
)

ISOO_COLUMNS = (
    "шифр",
    "тематика_обращения",
    *ISOO_BRANCH_COLUMNS,
    "всего_по_филиалам",
    *ISOO_CHANNEL_COLUMNS,
    "всего_по_каналам_поступления_обра",
    "date",
    "sys_section",
    "add_time",
)

ISOO_THEMES = (
    ("NEG_SAN", "Санитарное состояние", 0.15),
    ("NEG_COND", "Работа проводников", 0.12),
    ("NEG_COMF", "Уровень комфорта", 0.12),
    ("NEG_FOOD", "Питание и вагон-ресторан", 0.10),
    ("INFO", "Справочно-информационные вопросы", 0.27),
    ("THANK", "Благодарности", 0.14),
    ("OTHER", "Прочие обращения", 0.10),
)

ISOO_INCIDENT_MULTIPLIER = {
    "NEG_SAN": 2.15,
    "NEG_COND": 1.75,
    "NEG_COMF": 1.90,
    "NEG_FOOD": 1.65,
    "INFO": 1.08,
    "THANK": 0.78,
    "OTHER": 1.10,
}

MANUAL_COLUMNS = ("value_name", "value", "date", "sys_section", "metric", "add_time")
MANUAL_VALUE_NAMES = ("pass_count", "car_turnover", "cap_usage", "safety")


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def month_starts(start: date, end: date) -> Iterable[date]:
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def add_months(value: date, months: int) -> date:
    zero_based = value.year * 12 + value.month - 1 + months
    return date(zero_based // 12, zero_based % 12 + 1, 1)


def seasonal_factor(value: date) -> float:
    day = value.timetuple().tm_yday
    return 1 + 0.19 * math.sin(2 * math.pi * (day - 120) / 365.25)


def is_summer_pressure(value: date) -> bool:
    return date(2025, 6, 1) <= value <= date(2025, 8, 31)


def timestamp_after(value: date, days: int = 2) -> str:
    return datetime.combine(value + timedelta(days=days), time(8, 30)).isoformat(sep=" ")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate_stats(rng: random.Random) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for current_date in date_range(STATS_START, STATS_END):
        aggregates = {
            "plan": defaultdict(float),
            "fact": defaultdict(float),
        }
        for branch in BRANCHES:
            season = seasonal_factor(current_date)
            weekend = 1.08 if current_date.weekday() in {4, 5, 6} else 1.0
            plan_passengers = 32 * branch.weight * season * weekend
            surge = 0.0
            if is_summer_pressure(current_date):
                surge = 0.14 if branch.incident else 0.035

            for metric in ("plan", "fact"):
                if metric == "plan":
                    passengers = plan_passengers
                else:
                    passengers = plan_passengers * (
                        1 + branch.demand_bias + surge + rng.uniform(-0.035, 0.035)
                    )

                distance = branch.average_distance * (1 + rng.uniform(-0.018, 0.018))
                pass_turnover = passengers * distance
                occupancy = 0.71 + 0.045 * (season - 1)
                if metric == "fact":
                    occupancy += max(0.0, passengers / plan_passengers - 1) * 0.42
                    if branch.incident and is_summer_pressure(current_date):
                        occupancy += 0.035
                occupancy = clamp(occupancy, 0.62, 0.91)
                seat_turnover = pass_turnover / occupancy
                seats_per_car = 46 + rng.uniform(-3.5, 3.5)
                car_turnover = seat_turnover / seats_per_car

                values = {
                    "pass_turnover": round(pass_turnover, 2),
                    "pass_count": round(passengers, 2),
                    "car_turnover": round(car_turnover, 2),
                    "seat_turnover": round(seat_turnover, 2),
                }
                for name, value in values.items():
                    aggregates[metric][name] += value

                loaded_at = timestamp_after(current_date)
                rows.append(
                    {
                        **values,
                        "structure": branch.stats_name,
                        "date": current_date.isoformat(),
                        "sys_section": "day",
                        "metric": metric,
                        "add_time": loaded_at,
                        "updated_at": loaded_at,
                        "is_deleted": False,
                    }
                )

        for metric in ("plan", "fact"):
            values = {name: round(value, 2) for name, value in aggregates[metric].items()}
            loaded_at = timestamp_after(current_date)
            rows.append(
                {
                    **values,
                    "structure": "ФПК",
                    "date": current_date.isoformat(),
                    "sys_section": "day",
                    "metric": metric,
                    "add_time": loaded_at,
                    "updated_at": loaded_at,
                    "is_deleted": False,
                }
            )
    return rows


def _csi_plan(month: date) -> dict[str, float]:
    improvement = (month.year - 2023) * 0.35
    return {
        component: round(clamp(base + improvement, 0, 100), 2) for component, base in CSI_PLAN_BASE.items()
    }


def _csi_fact(
    rng: random.Random,
    *,
    month: date,
    branch: Branch,
    plan: dict[str, float],
) -> dict[str, float]:
    summer = month.month in {6, 7, 8}
    incident = branch.incident and month.year == 2025 and summer
    recovery_factor = (
        {9: 0.45, 10: 0.05}.get(month.month, 0.0) if branch.incident and month.year == 2025 else 0.0
    )
    recovery_bonus = (
        {11: 0.6, 12: 0.8}.get(month.month, 0.0) if branch.incident and month.year == 2025 else 0.0
    )
    result: dict[str, float] = {}
    for component, plan_value in plan.items():
        seasonal_penalty = 0.7 if summer and component != "безопасность" else 0.0
        incident_penalty = INCIDENT_PENALTY[component] if incident else 0.0
        residual_penalty = INCIDENT_PENALTY[component] * recovery_factor
        result[component] = round(
            clamp(
                plan_value
                + branch.quality_bias
                - seasonal_penalty
                - incident_penalty
                - residual_penalty
                + recovery_bonus
                + rng.uniform(-0.55, 0.55),
                0,
                100,
            ),
            2,
        )
    return result


def _csi_indexes(values: dict[str, float]) -> tuple[float, float]:
    satisfaction = round(sum(values.values()) / len(values), 2)
    loyalty = round(
        clamp(
            25
            + (satisfaction - 87)
            + 0.8 * (values["стоимость_поездки"] - 84)
            + 0.5 * (values["работа_проводников"] - 90)
            + 0.4 * (values["уровень_комфорта"] - 87),
            -100,
            100,
        ),
        2,
    )
    return satisfaction, loyalty


def _csi_add_time(month: date, row_type: str, *, old_plan: bool = False) -> str:
    if row_type == "plan":
        loaded = add_months(month, -1) + timedelta(days=4 if old_plan else 19)
    else:
        loaded = add_months(month, 1) + timedelta(days=2)
    return datetime.combine(loaded, time(8, 30)).isoformat(sep=" ")


def generate_csi(rng: random.Random) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in month_starts(HISTORY_START, HISTORY_END):
        branch_rows: dict[str, list[tuple[Branch, dict[str, float], float, float]]] = {
            "plan": [],
            "fact": [],
            "delta": [],
        }
        for branch in BRANCHES:
            plan = _csi_plan(month)
            fact = _csi_fact(rng, month=month, branch=branch, plan=plan)
            delta = {component: round(fact[component] - plan[component], 2) for component in CSI_COMPONENTS}
            plan_satisfaction, plan_loyalty = _csi_indexes(plan)
            fact_satisfaction, fact_loyalty = _csi_indexes(fact)
            indexes = {
                "plan": (plan_satisfaction, plan_loyalty),
                "fact": (fact_satisfaction, fact_loyalty),
                "delta": (
                    round(fact_satisfaction - plan_satisfaction, 2),
                    round(fact_loyalty - plan_loyalty, 2),
                ),
            }
            values_by_type = {"plan": plan, "fact": fact, "delta": delta}
            for row_type in ("plan", "fact", "delta"):
                values = values_by_type[row_type]
                satisfaction, loyalty = indexes[row_type]
                branch_rows[row_type].append((branch, values, satisfaction, loyalty))
                rows.append(
                    {
                        "type": row_type,
                        **values,
                        "date": month.isoformat(),
                        "индекс_удовлетворенности_пас": satisfaction,
                        "add_time": _csi_add_time(month, row_type),
                        "structure": branch.csi_name,
                        "индекс_потребительской_лояльност": loyalty,
                        "sys_section": "month",
                    }
                )

        for row_type in ("plan", "fact", "delta"):
            weighted = branch_rows[row_type]
            total_weight = sum(branch.weight for branch, *_ in weighted)
            values = {
                component: round(
                    sum(branch.weight * row[component] for branch, row, *_ in weighted) / total_weight,
                    2,
                )
                for component in CSI_COMPONENTS
            }
            satisfaction = round(
                sum(branch.weight * sat for branch, _, sat, _ in weighted) / total_weight,
                2,
            )
            loyalty = round(
                sum(branch.weight * nps for branch, _, _, nps in weighted) / total_weight,
                2,
            )
            if row_type == "plan" and month.year == 2025:
                revision_offsets = {
                    "работа_проводников": 0.5,
                    "санитарное_состояние": 0.7,
                    "уровень_комфорта": 0.6,
                    "услуги_вагона_ресторана": 0.6,
                }
                old_values = {
                    component: round(
                        value - revision_offsets.get(component, 0.0),
                        2,
                    )
                    for component, value in values.items()
                }
                old_satisfaction, old_loyalty = _csi_indexes(old_values)
                rows.append(
                    {
                        "type": row_type,
                        **old_values,
                        "date": month.isoformat(),
                        "индекс_удовлетворенности_пас": old_satisfaction,
                        "add_time": _csi_add_time(month, row_type, old_plan=True),
                        "structure": "ФПК",
                        "индекс_потребительской_лояльност": old_loyalty,
                        "sys_section": "month",
                    }
                )
            rows.append(
                {
                    "type": row_type,
                    **values,
                    "date": month.isoformat(),
                    "индекс_удовлетворенности_пас": satisfaction,
                    "add_time": _csi_add_time(month, row_type),
                    "structure": "ФПК",
                    "индекс_потребительской_лояльност": loyalty,
                    "sys_section": "month",
                }
            )
    return rows


def _isoo_profiles() -> dict[str, tuple[float, bool]]:
    return {branch.isoo_column: (branch.weight, branch.incident) for branch in BRANCHES}


def _allocate_channels(total: int, *, escalated: bool, rng: random.Random) -> list[int]:
    base = (0.32, 0.08, 0.34, 0.08) if escalated else (0.44, 0.10, 0.24, 0.08)
    shares = [share + rng.uniform(-0.012, 0.012) for share in base]
    first = [math.floor(total * share) for share in shares]
    return [*first, total - sum(first)]


def _build_isoo_row(
    *,
    code: str,
    theme: str,
    current_date: date,
    section: str,
    branch_values: dict[str, int],
    channel_values: dict[str, int] | None = None,
    escalated: bool = False,
    rng: random.Random | None = None,
) -> dict[str, object]:
    branch_total = sum(branch_values.values())
    assert channel_values is not None or rng is not None
    channels = channel_values or dict(
        zip(
            ISOO_CHANNEL_COLUMNS,
            _allocate_channels(branch_total, escalated=escalated, rng=rng),
            strict=True,
        )
    )
    assert sum(channels.values()) == branch_total
    return {
        "шифр": code,
        "тематика_обращения": theme,
        **branch_values,
        "всего_по_филиалам": branch_total,
        **channels,
        "всего_по_каналам_поступления_обра": branch_total,
        "date": current_date.isoformat(),
        "sys_section": section,
        "add_time": timestamp_after(current_date),
    }


def generate_isoo(rng: random.Random) -> list[dict[str, object]]:
    profiles = _isoo_profiles()
    daily_rows: list[dict[str, object]] = []
    for current_date in date_range(HISTORY_START, HISTORY_END):
        year_factor = {2023: 0.94, 2024: 0.97, 2025: 1.0}[current_date.year]
        details: list[dict[str, object]] = []
        for code, theme, theme_share in ISOO_THEMES:
            branch_values: dict[str, int] = {}
            for column in ISOO_BRANCH_COLUMNS:
                weight, incident_branch = profiles[column]
                modifier = 1.0
                if current_date.year == 2025 and current_date.month in {6, 7, 8} and incident_branch:
                    modifier = ISOO_INCIDENT_MULTIPLIER[code]
                elif current_date.year == 2025 and current_date.month >= 9 and incident_branch:
                    if code.startswith("NEG_"):
                        modifier = {9: 1.45, 10: 1.15, 11: 0.95}.get(current_date.month, 0.85)
                    elif code == "THANK":
                        modifier = {9: 0.90, 10: 1.00, 11: 1.10}.get(current_date.month, 1.18)

                expected = 20 * weight * seasonal_factor(current_date) * year_factor * theme_share * modifier
                branch_values[column] = max(0, round(expected + rng.uniform(-1.1, 1.1)))
            details.append(
                _build_isoo_row(
                    code=code,
                    theme=theme,
                    current_date=current_date,
                    section="day",
                    branch_values=branch_values,
                    escalated=(
                        current_date.year == 2025
                        and current_date.month in {6, 7, 8, 9}
                        and code.startswith("NEG_")
                    ),
                    rng=rng,
                )
            )

        total_values = {column: sum(int(row[column]) for row in details) for column in ISOO_BRANCH_COLUMNS}
        total_channel_values = {
            column: sum(int(row[column]) for row in details) for column in ISOO_CHANNEL_COLUMNS
        }
        daily_rows.extend(details)
        daily_rows.append(
            _build_isoo_row(
                code="TOTAL",
                theme="Поступило обращений всего",
                current_date=current_date,
                section="day",
                branch_values=total_values,
                channel_values=total_channel_values,
            )
        )

    numeric_columns = (
        *ISOO_BRANCH_COLUMNS,
        "всего_по_филиалам",
        *ISOO_CHANNEL_COLUMNS,
        "всего_по_каналам_поступления_обра",
    )
    monthly: dict[tuple[date, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in daily_rows:
        key = (
            month_start(date.fromisoformat(str(row["date"]))),
            str(row["шифр"]),
            str(row["тематика_обращения"]),
        )
        for column in numeric_columns:
            monthly[key][column] += int(row[column])

    monthly_rows = []
    for (period, code, theme), values in sorted(monthly.items()):
        monthly_rows.append(
            {
                "шифр": code,
                "тематика_обращения": theme,
                **values,
                "date": period.isoformat(),
                "sys_section": "month",
                "add_time": timestamp_after(add_months(period, 1)),
            }
        )
    return [*daily_rows, *monthly_rows]


def _monthly_stats(stats_rows: list[dict[str, object]]) -> dict[tuple[int, str], dict[str, float]]:
    aggregated: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in stats_rows:
        if row["structure"] != "ФПК":
            continue
        period = date.fromisoformat(str(row["date"]))
        key = (period.month, str(row["metric"]))
        for column in ("pass_count", "car_turnover", "pass_turnover", "seat_turnover"):
            aggregated[key][column] += float(row[column])
    return aggregated


def _manual_month_values(
    *,
    stats_monthly: dict[tuple[int, str], dict[str, float]],
    year: int,
    month: int,
    metric: str,
) -> tuple[dict[str, float], tuple[float, float]]:
    plan = stats_monthly[(month, "plan")]
    scale = {2023: 0.93, 2024: 0.97, 2025: 1.0, 2026: 1.035}[year]
    wave = 1 + 0.012 * math.sin((year * 12 + month) * 1.7) if year < 2025 else 1.0

    def historical(source: dict[str, float], column: str) -> float:
        return source[column] * scale * wave

    plan_passengers = historical(plan, "pass_count")
    plan_cars = historical(plan, "car_turnover") * 1_000
    plan_turnover = historical(plan, "pass_turnover")
    plan_seats = historical(plan, "seat_turnover")
    if year == 2026:
        plan_turnover *= 1.01
    plan_values = {
        "pass_count": plan_passengers,
        "car_turnover": plan_cars,
        "cap_usage": plan_turnover / plan_seats * 100,
        "safety": 0.0020,
    }
    if metric == "plan":
        return plan_values, (plan_turnover, plan_seats)

    if metric == "fact":
        if year == 2025:
            fact = stats_monthly[(month, "fact")]
            fact_passengers = historical(fact, "pass_count")
            fact_cars = historical(fact, "car_turnover") * 1_000
            fact_turnover = historical(fact, "pass_turnover")
            fact_seats = historical(fact, "seat_turnover")
        else:
            demand_gap = 0.004 + 0.009 * math.sin(year * 1.9 + month * 0.8)
            turnover_gap = demand_gap + 0.003 * math.cos(year + month)
            cap_gap = 0.20 + 0.30 * math.sin(year + month * 0.8)
            fact_passengers = plan_passengers * (1 + demand_gap)
            fact_turnover = plan_turnover * (1 + turnover_gap)
            fact_seats = fact_turnover / ((plan_values["cap_usage"] + cap_gap) / 100)
            fact_cars = plan_cars * (fact_seats / plan_seats + 0.002 * math.cos(year * 0.7 + month))
        fact_values = {
            "pass_count": fact_passengers,
            "car_turnover": fact_cars,
            "cap_usage": fact_turnover / fact_seats * 100,
            "safety": 0.0020
            + (0.000025 if year == 2025 and month in {6, 7, 8} else 0)
            + 0.000015 * math.sin(month),
        }
        return fact_values, (fact_turnover, fact_seats)

    expected_gap = 0.006 + 0.004 * math.sin(year * 2.1 + month * 0.7)
    capacity_gap = 0.003 + 0.003 * math.cos(year + month)
    if year == 2025 and month in {6, 7, 8}:
        expected_gap += 0.040
        capacity_gap += 0.012
    elif year == 2026 and month in {6, 7, 8}:
        expected_gap += 0.049
        capacity_gap += 0.014
    elif year == 2026:
        expected_gap += 0.006
        capacity_gap += 0.004
    forecast_turnover = plan_turnover * (1 + expected_gap + 0.002 * math.cos(year + month))
    forecast_seats = plan_seats * (1 + capacity_gap)
    forecast = {
        "pass_count": plan_passengers * (1 + expected_gap),
        "car_turnover": plan_cars * (1 + capacity_gap),
        "cap_usage": forecast_turnover / forecast_seats * 100,
        "safety": 0.0020 + 0.000012 * math.cos(year + month),
    }
    return forecast, (forecast_turnover, forecast_seats)


def generate_manual(stats_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stats_monthly = _monthly_stats(stats_rows)
    monthly: dict[tuple[date, str, str], float] = {}
    cap_parts: dict[tuple[date, str], tuple[float, float]] = {}
    for year in (2023, 2024, 2025, 2026):
        available_metrics = ("plan", "forecast") if year == 2026 else ("plan", "fact", "forecast")
        for month in range(1, 13):
            period = date(year, month, 1)
            for metric in available_metrics:
                values, parts = _manual_month_values(
                    stats_monthly=stats_monthly,
                    year=year,
                    month=month,
                    metric=metric,
                )
                cap_parts[(period, metric)] = parts
                for value_name, value in values.items():
                    monthly[(period, metric, value_name)] = value

    rows: list[dict[str, object]] = []

    def loaded_at(period: date, section: str, metric: str) -> str:
        if metric in {"plan", "forecast"}:
            loaded = date(
                period.year - 1,
                12,
                15 if metric == "plan" else 25,
            )
        else:
            months = {"month": 1, "quater": 3, "year": 12}[section]
            loaded = add_months(period, months) + timedelta(days=2)
        return datetime.combine(loaded, time(8, 30)).isoformat(sep=" ")

    def add_row(period: date, section: str, metric: str, value_name: str, value: float) -> None:
        rows.append(
            {
                "value_name": value_name,
                "value": round(value, 6 if value_name == "safety" else 4),
                "date": period.isoformat(),
                "sys_section": section,
                "metric": metric,
                "add_time": loaded_at(period, section, metric),
            }
        )

    def aggregate(
        periods: list[date],
        metric: str,
        value_name: str,
    ) -> float:
        if value_name == "cap_usage":
            turnover = sum(cap_parts[(period, metric)][0] for period in periods)
            seats = sum(cap_parts[(period, metric)][1] for period in periods)
            return turnover / seats * 100
        values = [monthly[(period, metric, value_name)] for period in periods]
        if value_name in {"pass_count", "car_turnover"}:
            return sum(values)
        return sum(values) / len(values)

    for (period, metric, value_name), value in sorted(monthly.items()):
        add_row(period, "month", metric, value_name, value)

    for year in (2023, 2024, 2025, 2026):
        available_metrics = (
            ("plan", "forecast")
            if year == 2026
            else (
                "plan",
                "fact",
                "forecast",
            )
        )
        for quarter_month in (1, 4, 7, 10):
            quarter = date(year, quarter_month, 1)
            periods = [date(year, quarter_month + offset, 1) for offset in range(3)]
            for metric in available_metrics:
                for value_name in MANUAL_VALUE_NAMES:
                    # `quater` is retained because it is the source-system value.
                    add_row(
                        quarter,
                        "quater",
                        metric,
                        value_name,
                        aggregate(periods, metric, value_name),
                    )

        for metric in available_metrics:
            for value_name in MANUAL_VALUE_NAMES:
                periods = [date(year, month, 1) for month in range(1, 13)]
                add_row(
                    date(year, 1, 1),
                    "year",
                    metric,
                    value_name,
                    aggregate(periods, metric, value_name),
                )
    return rows


def validate(
    stats: list[dict[str, object]],
    csi: list[dict[str, object]],
    isoo: list[dict[str, object]],
    manual: list[dict[str, object]],
) -> None:
    datasets = {
        "stat_stats": stats,
        "stat_csi": csi,
        "stat_isoo": isoo,
        "stat_manual": manual,
    }
    for name, rows in datasets.items():
        assert rows, f"{name} is empty"
        assert len(rows) <= 10_000, f"{name} exceeds 10,000 rows"

    stats_keys = [(row["date"], row["structure"], row["metric"]) for row in stats]
    assert len(stats_keys) == len(set(stats_keys))
    assert all(39 <= float(row["seat_turnover"]) / float(row["car_turnover"]) <= 53 for row in stats)
    stats_lookup = {(row["date"], row["structure"], row["metric"]): row for row in stats}
    for current_date in date_range(STATS_START, STATS_END):
        for metric in ("plan", "fact"):
            fpk = stats_lookup[(current_date.isoformat(), "ФПК", metric)]
            for column in ("pass_turnover", "pass_count", "car_turnover", "seat_turnover"):
                branch_sum = round(
                    sum(
                        float(stats_lookup[(current_date.isoformat(), branch.stats_name, metric)][column])
                        for branch in BRANCHES
                    ),
                    2,
                )
                assert abs(float(fpk[column]) - branch_sum) < 0.011

    incident_stats = {branch.stats_name for branch in BRANCHES if branch.incident}
    summer_stats = [row for row in stats if "2025-07-01" <= str(row["date"]) <= "2025-08-31"]

    def passenger_ratio(structures: set[str]) -> float:
        fact = sum(
            float(row["pass_count"])
            for row in summer_stats
            if row["structure"] in structures and row["metric"] == "fact"
        )
        plan = sum(
            float(row["pass_count"])
            for row in summer_stats
            if row["structure"] in structures and row["metric"] == "plan"
        )
        return fact / plan

    assert passenger_ratio(incident_stats) > 1.12
    assert passenger_ratio({"МОСК."}) < 1.08

    csi_keys = [
        (
            row["date"],
            row["structure"],
            row["type"],
            row["sys_section"],
            row["add_time"],
        )
        for row in csi
    ]
    assert len(csi_keys) == len(set(csi_keys))
    logical_csi_keys = [(row["date"], row["structure"], row["type"], row["sys_section"]) for row in csi]
    assert len(logical_csi_keys) - len(set(logical_csi_keys)) == 12
    csi_versions: dict[
        tuple[object, object, object, object],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in csi:
        csi_versions[
            (
                row["date"],
                row["structure"],
                row["type"],
                row["sys_section"],
            )
        ].append(row)
    revision_groups = [
        (key, sorted(rows, key=lambda row: str(row["add_time"])))
        for key, rows in csi_versions.items()
        if len(rows) > 1
    ]
    assert len(revision_groups) == 12
    expected_changes = {
        "работа_проводников",
        "санитарное_состояние",
        "уровень_комфорта",
        "услуги_вагона_ресторана",
        "индекс_удовлетворенности_пас",
        "индекс_потребительской_лояльност",
    }
    for key, (old_plan, latest_plan) in revision_groups:
        assert key[1:] == ("ФПК", "plan", "month")
        assert str(key[0]).startswith("2025-")
        changed = {
            column
            for column in (
                *CSI_COMPONENTS,
                "индекс_удовлетворенности_пас",
                "индекс_потребительской_лояльност",
            )
            if old_plan[column] != latest_plan[column]
        }
        assert changed == expected_changes
    csi_lookup: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in csi:
        key = (row["date"], row["structure"], row["type"])
        if key not in csi_lookup or str(row["add_time"]) > str(csi_lookup[key]["add_time"]):
            csi_lookup[key] = row
    for period, structure, row_type in csi_lookup:
        if row_type != "delta":
            continue
        delta = csi_lookup[(period, structure, "delta")]
        plan = csi_lookup[(period, structure, "plan")]
        fact = csi_lookup[(period, structure, "fact")]
        for column in (
            *CSI_COMPONENTS,
            "индекс_удовлетворенности_пас",
            "индекс_потребительской_лояльност",
        ):
            assert abs(float(delta[column]) - round(float(fact[column]) - float(plan[column]), 2)) <= 0.02

    incident_csi = {branch.csi_name for branch in BRANCHES if branch.incident}
    summer_csi = [
        float(row["индекс_удовлетворенности_пас"])
        for row in csi
        if row["type"] == "delta"
        and row["structure"] in incident_csi
        and row["date"] in {"2025-06-01", "2025-07-01", "2025-08-01"}
    ]
    moscow_csi = [
        float(row["индекс_удовлетворенности_пас"])
        for row in csi
        if row["type"] == "delta"
        and row["structure"] == "МОСК"
        and row["date"] in {"2025-06-01", "2025-07-01", "2025-08-01"}
    ]

    def incident_csi_for(period: str) -> float:
        values = [
            float(row["индекс_удовлетворенности_пас"])
            for row in csi
            if row["type"] == "delta" and row["structure"] in incident_csi and row["date"] == period
        ]
        return sum(values) / len(values)

    assert sum(summer_csi) / len(summer_csi) < -3.5
    assert sum(moscow_csi) / len(moscow_csi) > -1.0
    september_csi = incident_csi_for("2025-09-01")
    october_csi = incident_csi_for("2025-10-01")
    november_csi = incident_csi_for("2025-11-01")
    assert -2.8 < september_csi < -1.0
    assert october_csi > september_csi + 0.6
    assert november_csi > october_csi + 0.3
    assert november_csi > -0.5
    assert (
        min(
            float(row["безопасность"])
            for row in csi
            if row["type"] == "fact" and row["date"].startswith("2025")
        )
        > 90
    )

    isoo_keys = [(row["date"], row["sys_section"], row["тематика_обращения"]) for row in isoo]
    assert len(isoo_keys) == len(set(isoo_keys))
    for row in isoo:
        branch_total = sum(int(row[column]) for column in ISOO_BRANCH_COLUMNS)
        channel_total = sum(int(row[column]) for column in ISOO_CHANNEL_COLUMNS)
        assert branch_total == int(row["всего_по_филиалам"])
        assert channel_total == int(row["всего_по_каналам_поступления_обра"])
        assert branch_total == channel_total
    grouped_isoo: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in isoo:
        grouped_isoo[(str(row["date"]), str(row["sys_section"]))].append(row)
    for rows in grouped_isoo.values():
        total = next(row for row in rows if row["шифр"] == "TOTAL")
        details = [row for row in rows if row["шифр"] != "TOTAL"]
        for column in (*ISOO_BRANCH_COLUMNS, *ISOO_CHANNEL_COLUMNS):
            assert int(total[column]) == sum(int(row[column]) for row in details)

    isoo_numeric_columns = (
        *ISOO_BRANCH_COLUMNS,
        "всего_по_филиалам",
        *ISOO_CHANNEL_COLUMNS,
        "всего_по_каналам_поступления_обра",
    )
    daily_isoo: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    monthly_isoo = {}
    for row in isoo:
        period = month_start(date.fromisoformat(str(row["date"]))).isoformat()
        key = (period, str(row["шифр"]))
        if row["sys_section"] == "day":
            for column in isoo_numeric_columns:
                daily_isoo[key][column] += int(row[column])
        else:
            monthly_isoo[key] = row
    for key, values in daily_isoo.items():
        for column in isoo_numeric_columns:
            assert int(monthly_isoo[key][column]) == values[column]

    negative_codes = {code for code, _, _ in ISOO_THEMES if code.startswith("NEG_")}
    incident_isoo = {branch.isoo_column for branch in BRANCHES if branch.incident}

    def isoo_volume(periods: set[str]) -> float:
        return sum(
            float(row[column])
            for row in isoo
            if row["sys_section"] == "month" and row["date"] in periods and row["шифр"] in negative_codes
            for column in incident_isoo
        )

    pre_periods = {"2025-03-01", "2025-04-01", "2025-05-01"}
    peak_periods = {"2025-06-01", "2025-07-01", "2025-08-01"}
    post_periods = {"2025-09-01", "2025-10-01", "2025-11-01"}
    assert isoo_volume(peak_periods) > isoo_volume(pre_periods) * 1.8
    assert isoo_volume(post_periods) < isoo_volume(peak_periods) * 0.85
    assert isoo_volume({"2025-09-01"}) > isoo_volume({"2025-10-01"}) > isoo_volume({"2025-11-01"})

    def escalated_share(periods: set[str]) -> float:
        selected = [
            row
            for row in isoo
            if row["sys_section"] == "month" and row["date"] in periods and row["шифр"] in negative_codes
        ]
        total = sum(float(row["всего_по_каналам_поступления_обра"]) for row in selected)
        escalated = sum(
            float(row["портал_генерального_директора_оао"]) + float(row["почта_генерального_директора_ао_ф"])
            for row in selected
        )
        return escalated / total

    assert escalated_share(peak_periods) > escalated_share(pre_periods) + 0.08

    manual_keys = [(row["date"], row["sys_section"], row["value_name"], row["metric"]) for row in manual]
    assert len(manual_keys) == len(set(manual_keys))
    manual_lookup = {
        (row["date"], row["sys_section"], row["value_name"], row["metric"]): float(row["value"])
        for row in manual
    }
    for month in range(1, 13):
        period = date(2025, month, 1).isoformat()
        plan_safety = manual_lookup[(period, "month", "safety", "plan")]
        fact_safety = manual_lookup[(period, "month", "safety", "fact")]
        assert abs(fact_safety / plan_safety - 1) <= 0.025
    for year in (2023, 2024):
        for month in range(1, 13):
            period = date(year, month, 1).isoformat()
            plan_passengers = manual_lookup[(period, "month", "pass_count", "plan")]
            fact_passengers = manual_lookup[(period, "month", "pass_count", "fact")]
            plan_usage = manual_lookup[(period, "month", "cap_usage", "plan")]
            fact_usage = manual_lookup[(period, "month", "cap_usage", "fact")]
            assert abs(fact_passengers / plan_passengers - 1) < 0.02
            assert abs(fact_usage - plan_usage) < 0.8
    for row in manual:
        period = date.fromisoformat(str(row["date"]))
        loaded = datetime.fromisoformat(str(row["add_time"])).date()
        if row["metric"] in {"plan", "forecast"}:
            assert loaded < period
        else:
            months_after = {"month": 1, "quater": 3, "year": 12}[str(row["sys_section"])]
            assert (
                add_months(period, months_after)
                < loaded
                <= add_months(period, months_after) + timedelta(days=5)
            )
        if row["sys_section"] not in {"quater", "year"}:
            continue
        count = 3 if row["sys_section"] == "quater" else 12
        months = [add_months(period, offset) for offset in range(count)]
        values = [
            manual_lookup[(month.isoformat(), "month", row["value_name"], row["metric"])] for month in months
        ]
        if row["value_name"] == "cap_usage":
            assert min(values) <= float(row["value"]) <= max(values)
        else:
            expected = (
                sum(values)
                if row["value_name"] in {"pass_count", "car_turnover"}
                else sum(values) / len(values)
            )
            assert math.isclose(
                float(row["value"]),
                expected,
                rel_tol=1e-9,
                abs_tol=0.001,
            )

    for value_name in ("pass_count", "cap_usage"):
        plan_error = 0.0
        forecast_error = 0.0
        for period in peak_periods:
            fact = manual_lookup[(period, "month", value_name, "fact")]
            plan_error += abs(manual_lookup[(period, "month", value_name, "plan")] - fact)
            forecast_error += abs(manual_lookup[(period, "month", value_name, "forecast")] - fact)
        assert 0.15 < forecast_error / plan_error < 0.65

    stats_monthly = _monthly_stats(stats)
    forecast_before = _manual_month_values(
        stats_monthly=stats_monthly,
        year=2025,
        month=7,
        metric="forecast",
    )
    changed_stats = {key: dict(values) for key, values in stats_monthly.items()}
    changed_stats[(7, "fact")]["pass_count"] *= 10
    changed_stats[(7, "fact")]["pass_turnover"] *= 10
    forecast_after = _manual_month_values(
        stats_monthly=changed_stats,
        year=2025,
        month=7,
        metric="forecast",
    )
    assert forecast_before == forecast_after


def generate(output_dir: Path, *, seed: int = SEED) -> dict[str, int]:
    rng = random.Random(seed)
    stats = generate_stats(rng)
    csi = generate_csi(rng)
    isoo = generate_isoo(rng)
    manual = generate_manual(stats)
    validate(stats, csi, isoo, manual)

    write_csv(output_dir / "stg_stat_stats.csv", STATS_COLUMNS, stats)
    write_csv(output_dir / "stg_stat_csi.csv", CSI_COLUMNS, csi)
    write_csv(output_dir / "stg_stat_isoo.csv", ISOO_COLUMNS, isoo)
    write_csv(output_dir / "stg_stat_manual.csv", MANUAL_COLUMNS, manual)
    return {
        "stat_stats": len(stats),
        "stat_csi": len(csi),
        "stat_isoo": len(isoo),
        "stat_manual": len(manual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic FPK demo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("generated"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    counts = generate(args.output_dir, seed=args.seed)
    for table, count in counts.items():
        print(f"{table}: {count:,} rows")


if __name__ == "__main__":
    main()
