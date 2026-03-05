from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query_text: str
    required_phrases: list[str]
    notes: str
    is_answerable: bool


def load_manual_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_query_cases() -> list[QueryCase]:
    return [
        QueryCase(
            query_id="Q1",
            query_text="When can you use the bypass key and what timer should you arm right after?",
            required_phrases=["bypass key is sealed", "arm the ninety second timer"],
            notes="Power section links a condition and immediate follow-up action.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q2",
            query_text="What airflow threshold triggers action and which fan setting is required?",
            required_phrases=["airflow drops below thirty cfm", "set fan speed to sixty percent"],
            notes="Cooling section ties signal threshold to a specific control value.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q3",
            query_text="If the coolant pump fails, what reset lever should be used?",
            required_phrases=["coolant pump fails", "reset lever is marked j 17"],
            notes="Boundary-sensitive pair used in chunking demonstrations.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q4",
            query_text="Which contamination indicator should be treated as critical and where is it recorded?",
            required_phrases=["sweet almond smell means contamination", "log it in bay ledger seven"],
            notes="Fuel section links smell cue and documentation location.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q5",
            query_text="Which command opens the panel, and what checksum ending confirms success?",
            required_phrases=["run command open panel delta", "checksum should end with 44af"],
            notes="Software section combines command and validation output.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q6",
            query_text="What light pattern means hold position and who can authorize movement?",
            required_phrases=["triple amber flash means hold position", "movement requires supervisor code"],
            notes="Safety section links indicator and authorization rule.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q7",
            query_text="For calibration, which port comes first and what drift limit is acceptable?",
            required_phrases=["connect probe to port c first", "drift must stay below zero point three"],
            notes="Calibration section includes sequence and numeric acceptance threshold.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q8",
            query_text="What defines the quiet window and what task is banned then?",
            required_phrases=["quiet window starts at 23 00", "do not purge the pressure lines"],
            notes="Operations section combines schedule and prohibited task.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q9",
            query_text="What phrase confirms handoff and where should backup notes be written?",
            required_phrases=["say baton accepted on channel six", "write backup note on orange card"],
            notes="Handoff section combines spoken confirmation and physical note target.",
            is_answerable=True,
        ),
        QueryCase(
            query_id="Q10",
            query_text="What is the backup battery cycle time and what seal code is required?",
            required_phrases=["backup battery cycle is twelve hours", "seal code is violet nine"],
            notes="Intentionally unanswerable query.",
            is_answerable=False,
        ),
        QueryCase(
            query_id="Q11",
            query_text="Which thermal vent alarm color is critical and which dockmaster channel should be used?",
            required_phrases=["thermal vent alarm is cobalt", "escalate to dockmaster channel nine"],
            notes="Intentionally unanswerable query.",
            is_answerable=False,
        ),
        QueryCase(
            query_id="Q12",
            query_text="If the coolant pump fails, which reset lever is used and which timer should be armed?",
            required_phrases=[
                "coolant pump fails",
                "reset lever is marked j 17",
                "arm the ninety second timer",
            ],
            notes="Boundary-sensitive multi-phrase query.",
            is_answerable=True,
        ),
    ]
