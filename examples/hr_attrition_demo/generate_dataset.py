from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


COLUMNS = (
    "employee_id",
    "snapshot_date",
    "hire_date",
    "termination_date",
    "attrition_90d",
    "contact_center",
    "project",
    "team",
    "tenure_months",
    "shift_type",
    "workload_pct",
    "overtime_hours_30d",
    "absence_days_90d",
    "late_shifts_90d",
    "schedule_adherence_pct",
    "productivity_score",
    "quality_score",
    "csat_score",
    "engagement_score",
    "manager_changed_90d",
    "training_hours_90d",
    "exit_reason",
)

DEMO_CUTOFF = date(2026, 6, 30)


def _clamp(value: float, low: float, high: float) -> float:
    return round(min(high, max(low, value)), 2)


def _month_starts(start: date, end: date) -> list[date]:
    months: list[date] = []
    current = start
    while current <= end:
        months.append(current)
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return months


def _exit_reason(
    rng: random.Random,
    *,
    engagement: float,
    overtime: float,
    quality: float,
) -> str:
    if engagement < 55:
        return "Низкая вовлечённость"
    if overtime > 24:
        return "Высокая нагрузка"
    if quality < 70:
        return "Несоответствие требованиям"
    return rng.choice(("Личные причины", "Смена работодателя", "Смена графика"))


def _make_row(
    rng: random.Random,
    *,
    employee_number: int,
    snapshot: date,
    historical: bool,
) -> dict[str, object]:
    center = rng.choice(("Москва", "Казань", "Омск"))
    project = rng.choice(("Поддержка", "Продажи", "Удержание"))
    shift = rng.choices(("Дневная", "Ночная", "Смешанная"), weights=(6, 2, 3))[0]
    manager_changed = int(rng.random() < 0.14)
    tenure = rng.randint(1, 60)
    workload = _clamp(rng.gauss(87, 13) + (6 if project == "Продажи" else 0), 50, 135)
    overtime = _clamp(rng.gauss(11, 8) + max(0, workload - 100) * 0.45, 0, 55)
    absence = min(14, int(rng.expovariate(0.38)))
    late = min(12, int(rng.expovariate(0.55)))
    adherence = _clamp(rng.gauss(94, 4) - absence * 1.4 - late * 0.8, 55, 100)
    productivity = _clamp(rng.gauss(81, 9) - max(0, workload - 108) * 0.35, 35, 100)
    quality = _clamp(rng.gauss(86, 7) - absence * 0.35, 45, 100)
    csat = _clamp(rng.gauss(4.35, 0.35) - max(0, 78 - quality) * 0.018, 1, 5)
    engagement = _clamp(
        rng.gauss(75, 12) - overtime * 0.38 - manager_changed * 10 - absence * 0.55,
        20,
        100,
    )
    training = _clamp(rng.gauss(7, 5), 0, 30)

    logit = (
        -2.75
        + max(0, overtime - 12) * 0.055
        + absence * 0.11
        + late * 0.09
        + max(0, 72 - engagement) * 0.035
        + max(0, 78 - quality) * 0.025
        + manager_changed * 0.42
        + (0.25 if shift == "Ночная" else 0)
        + (0.22 if tenure <= 4 else 0)
        + (0.18 if snapshot.month in {1, 8, 9} else 0)
    )
    attrition = int(historical and rng.random() < 1 / (1 + math.exp(-logit)))
    termination = snapshot + timedelta(days=rng.randint(7, 89)) if attrition else None

    return {
        "employee_id": f"SYN-{employee_number:05d}",
        "snapshot_date": snapshot.isoformat(),
        "hire_date": (snapshot - timedelta(days=tenure * 30)).isoformat(),
        "termination_date": termination.isoformat() if termination else "",
        "attrition_90d": attrition if historical else "",
        "contact_center": center,
        "project": project,
        "team": f"Команда {rng.randint(1, 12):02d}",
        "tenure_months": tenure,
        "shift_type": shift,
        "workload_pct": workload,
        "overtime_hours_30d": overtime,
        "absence_days_90d": absence,
        "late_shifts_90d": late,
        "schedule_adherence_pct": adherence,
        "productivity_score": productivity,
        "quality_score": quality,
        "csat_score": csat,
        "engagement_score": engagement,
        "manager_changed_90d": manager_changed,
        "training_hours_90d": training,
        "exit_reason": (
            _exit_reason(rng, engagement=engagement, overtime=overtime, quality=quality)
            if attrition
            else ""
        ),
    }


def generate_rows(*, row_count: int = 3600, seed: int = 20260720) -> list[dict[str, object]]:
    if row_count < 3000:
        raise ValueError("HR attrition demo requires at least 3000 rows.")

    rng = random.Random(seed)
    current_count = round(row_count * 2 / 9)
    historical_count = row_count - current_count
    history_months = _month_starts(date(2024, 1, 1), date(2026, 3, 1))

    rows = [
        _make_row(
            rng,
            employee_number=index + 1,
            snapshot=rng.choice(history_months) + timedelta(days=rng.randint(0, 20)),
            historical=True,
        )
        for index in range(historical_count)
    ]
    rows.extend(
        _make_row(
            rng,
            employee_number=historical_count + index + 1,
            snapshot=DEMO_CUTOFF,
            historical=False,
        )
        for index in range(current_count)
    )
    return rows


def write_dataset(path: Path, *, row_count: int = 3600, seed: int = 20260720) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(generate_rows(row_count=row_count, seed=seed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic HR attrition demo data.")
    parser.add_argument("--rows", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("dataset.csv"))
    args = parser.parse_args()
    write_dataset(args.output, row_count=args.rows, seed=args.seed)
    print(f"Generated {args.rows} synthetic rows: {args.output}")


if __name__ == "__main__":
    main()
