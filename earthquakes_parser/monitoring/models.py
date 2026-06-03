"""Data models for pipeline monitoring metrics."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class PipelineSnapshot:
    """Snapshot of pipeline processing state at a given moment."""

    period_hours: int
    search_results_total: int
    search_results_by_status: Dict[str, int]
    parsed_content_total: int
    ingested_total: int
    pipeline_runs_total: int
    pipeline_runs_in_period: int
    error_runs_in_period: int
    last_run_at: Optional[datetime]
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
