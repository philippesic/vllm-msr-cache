from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CORE = ("gpu-only", "native-arc", "wait-hint")
EXTERNAL = ("lmcache",)
MODELS = ("qwen35-4b-int4", "gemma4-12b-int4")
TASKS = (
    "C1-1",
    "C1-2",
    "C1-3",
    "C1-4",
    "C2-1",
    "C2-2",
    "C3-1",
    "C3-2",
    "C4-1",
    "C4-2",
    "C5-1",
    "B1-1",
    "B1-2",
)
EXTERNAL_TASKS = ("C1-1", "C2-1", "C3-1", "C4-1", "C5-1", "B1-1")


@dataclass(frozen=True)
class Cell:
    model: str
    system: str
    case: str
    block: int
    kind: str
    seconds: int

    @property
    def identity(self) -> str:
        return f"{self.kind}/{self.model}/{self.case}/{self.system}/{self.block}"


def cells(profile: str) -> tuple[Cell, ...]:
    result: list[Cell] = []
    if profile == "functional":
        return (
            Cell(
                "dense-fixture", "wait-hint", "native-functional", 0, "functional", 600
            ),
        )
    if profile == "compare":
        return tuple(
            Cell(MODELS[0], arm, task, 0, "application", 120)
            for arm in CORE + EXTERNAL
            for task in ("V1", "V2", "V3")
        )
    if profile != "release":
        raise ValueError("unknown profile")
    for block in range(2):
        arms = CORE + EXTERNAL if block == 0 else tuple(reversed(CORE + EXTERNAL))
        for model in MODELS:
            for arm in arms:
                tasks = TASKS if arm in CORE else EXTERNAL_TASKS
                result.extend(
                    Cell(model, arm, task, block, "application", 240) for task in tasks
                )
            for case in ("T1", "T2", "T3"):
                result.extend(
                    Cell(model, arm, case, block, "trace", 180)
                    for arm in arms
                    if arm in CORE
                )
            result.append(Cell(model, "previous-or-base", "R1", block, "trace", 180))
        for case, arm in (
            ("O1", "no-registration"),
            ("O2", "registration-only"),
            ("A1", "native-arc"),
            ("S1", "injected-delay"),
        ):
            result.append(Cell(MODELS[0], arm, case, block, "trace", 180))
        result.extend(
            Cell("dense-fixture", arm, "D1", block, "trace", 180)
            for arm in ("wait-hint", "previous-or-base")
        )
    for model, seconds in zip(MODELS, (7200, 3600), strict=True):
        result.extend(
            Cell(model, arm, "sustained", 0, "soak", seconds)
            for arm in ("native-arc", "wait-hint")
        )
    return tuple(result)


def describe(profile: str) -> dict[str, Any]:
    scheduled = cells(profile)
    return {
        "schema": "benchmark-plan-v1",
        "profile": profile,
        "status": "unqualified_schedule",
        "budget_seconds": {"functional": 900, "compare": 2700, "release": 108000}[
            profile
        ],
        "repetitions": 2 if profile == "release" else 1,
        "counts": {
            kind: sum(cell.kind == kind for cell in scheduled)
            for kind in ("functional", "application", "trace", "soak")
        },
        "cell_timeout_sum_seconds": sum(cell.seconds for cell in scheduled),
        "admission_required": [
            "pinned_assets",
            "system_compatibility",
            "quality_rubrics",
            "activity_contracts",
            "fault_receipts",
            "pilot_cost",
        ],
        "cells": [{"id": cell.identity, **asdict(cell)} for cell in scheduled],
    }
