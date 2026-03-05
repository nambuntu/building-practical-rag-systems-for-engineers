from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query_text: str
    required_phrases: list[str]
    notes: str


def load_manual_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_query_cases() -> list[QueryCase]:
    return [
        QueryCase(
            query_id="Q1",
            query_text="When can you use the bypass key and what timer should be armed after using it?",
            required_phrases=["bypass key is sealed", "arm the ninety second timer"],
            notes="Power section ties a condition to an immediate timing step.",
        ),
        QueryCase(
            query_id="Q2",
            query_text="If airflow falls below safe range, what threshold causes an alarm and what fan speed is recommended?",
            required_phrases=["airflow drops below thirty cfm", "set fan speed to sixty percent"],
            notes="Cooling section bundles threshold + response action.",
        ),
        QueryCase(
            query_id="Q3",
            query_text="If the coolant pump fails, what is the first action and which lever resets it?",
            required_phrases=["coolant pump fails", "reset lever is marked j 17"],
            notes="Designed to straddle fixed boundaries in small chunks.",
        ),
        QueryCase(
            query_id="Q4",
            query_text="During fuel checks, what smell indicates contamination and where should it be logged?",
            required_phrases=["sweet almond smell means contamination", "log it in bay ledger seven"],
            notes="Fuel section joins diagnostic cue and logging location.",
        ),
        QueryCase(
            query_id="Q5",
            query_text="For diagnostics, which command opens the panel and which checksum confirms success?",
            required_phrases=["run command open panel delta", "checksum should end with 44af"],
            notes="Software section mixes command and checksum fact.",
        ),
        QueryCase(
            query_id="Q6",
            query_text="In emergency mode, which light pattern means hold position and who must authorize movement?",
            required_phrases=["triple amber flash means hold position", "movement requires supervisor code"],
            notes="Safety section includes status signal and authorization rule.",
        ),
        QueryCase(
            query_id="Q7",
            query_text="When calibrating sensors, which port should be used first and what drift limit is acceptable?",
            required_phrases=["connect probe to port c first", "drift must stay below zero point three"],
            notes="Calibration section links sequence and acceptance threshold.",
        ),
        QueryCase(
            query_id="Q8",
            query_text="For overnight maintenance, what time is the quiet window and what task is banned during that window?",
            required_phrases=["quiet window starts at 23 00", "do not purge the pressure lines"],
            notes="Operations section anchors schedule and prohibited task.",
        ),
        QueryCase(
            query_id="Q9",
            query_text="During handoff, what phrase confirms ownership transfer and where is the backup note written?",
            required_phrases=["say baton accepted on channel six", "write backup note on orange card"],
            notes="Handoff section uses a spoken confirmation and physical artifact.",
        ),
    ]
