"""Metrics collector: queries Supabase to build a PipelineSnapshot."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from earthquakes_parser.monitoring.models import PipelineSnapshot
from earthquakes_parser.storage.supabase import SupabaseDB

_STATUSES = ["pending", "downloaded", "parsed", "analyzed", "ingested", "failed"]


class MetricsCollector:
    """Collects pipeline metrics from Supabase tables."""

    def __init__(self, db: SupabaseDB) -> None:
        """Initialize collector with a Supabase database client.

        Args:
            db: Configured SupabaseDB instance.
        """
        self.db = db

    def collect(self, period_hours: int = 24) -> PipelineSnapshot:
        """Query Supabase tables and return a current PipelineSnapshot.

        Args:
            period_hours: Hours to look back for pipeline run statistics.

        Returns:
            PipelineSnapshot populated with the latest metrics.
        """
        search_by_status = self._count_search_results_by_status()
        search_total: int = sum(search_by_status.values())
        parsed_total = self._count_parsed_content()
        ingested_total = search_by_status.get("ingested", 0)
        runs_total, runs_period, errors_period, last_run = self._count_pipeline_runs(
            period_hours
        )

        return PipelineSnapshot(
            period_hours=period_hours,
            search_results_total=search_total,
            search_results_by_status=search_by_status,
            parsed_content_total=parsed_total,
            ingested_total=ingested_total,
            pipeline_runs_total=runs_total,
            pipeline_runs_in_period=runs_period,
            error_runs_in_period=errors_period,
            last_run_at=last_run,
        )

    def _count_search_results_by_status(self) -> Dict[str, int]:
        """Return search result counts grouped by processing status.

        Returns:
            Dict mapping each status name to its record count.
        """
        df = self.db.select("search_results", columns="status")
        if df.empty:
            return {s: 0 for s in _STATUSES}
        counts = df["status"].value_counts()
        return {s: int(counts.get(s, 0)) for s in _STATUSES}

    def _count_parsed_content(self) -> int:
        """Return the total number of rows in the parsed_content table.

        Returns:
            Row count as an integer.
        """
        df = self.db.select("parsed_content", columns="id")
        return len(df)

    def _count_pipeline_runs(
        self, period_hours: int
    ) -> Tuple[int, int, int, Optional[datetime]]:
        """Return pipeline run statistics from the pipeline_runs table.

        Args:
            period_hours: Time window in hours for recent run counts.

        Returns:
            Tuple of (total, runs_in_period, errors_in_period, last_run_at).
        """
        df = self.db.select("pipeline_runs", columns="id,status,started_at")
        if df.empty:
            return 0, 0, 0, None

        total: int = len(df)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=period_hours)

        df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
        recent = df[df["started_at"] >= cutoff]
        runs_in_period: int = len(recent)
        errors_in_period: int = int(
            (recent["status"] == "error").sum() if not recent.empty else 0
        )

        last_ts = df["started_at"].max()
        last_run: Optional[datetime] = (
            last_ts.to_pydatetime() if pd.notna(last_ts) else None
        )

        return total, runs_in_period, errors_in_period, last_run
