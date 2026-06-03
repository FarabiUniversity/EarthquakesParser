"""Reporter: render pipeline metrics as Rich tables and HTML reports."""

import base64
import io
import os
from typing import Optional

from earthquakes_parser.monitoring.models import PipelineSnapshot

_STATUS_ORDER = [
    "pending",
    "downloaded",
    "parsed",
    "analyzed",
    "ingested",
    "failed",
]


class PipelineReporter:
    """Formats and outputs PipelineSnapshot metrics."""

    def print_table(self, snapshot: PipelineSnapshot) -> None:
        """Print a Rich summary table of pipeline state to the terminal.

        Args:
            snapshot: PipelineSnapshot containing current metrics.

        Raises:
            ImportError: If the rich package is not installed.
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
        except ImportError as exc:
            raise ImportError(
                "rich is required. " "Install with: uv pip install -e '.[monitoring]'"
            ) from exc

        console = Console()
        ts = snapshot.collected_at.strftime("%Y-%m-%d %H:%M UTC")
        console.print(Panel(f"Pipeline Monitor — {ts}", style="bold blue"))

        sr_table = Table(title="Search Results by Status", show_lines=True)
        sr_table.add_column("Status", style="cyan", min_width=12)
        sr_table.add_column("Count", justify="right", style="magenta")
        for status in _STATUS_ORDER:
            count = snapshot.search_results_by_status.get(status, 0)
            label = f"[red]{status}[/red]" if status == "failed" else status
            sr_table.add_row(label, str(count))
        sr_table.add_row("[bold]Total[/bold]", str(snapshot.search_results_total))
        console.print(sr_table)

        period_label = f"last {snapshot.period_hours}h"
        runs_table = Table(title="Pipeline Runs", show_lines=True)
        runs_table.add_column("Metric", style="cyan", min_width=24)
        runs_table.add_column("Value", justify="right", style="magenta")
        runs_table.add_row("Total runs", str(snapshot.pipeline_runs_total))
        runs_table.add_row(
            f"Runs ({period_label})", str(snapshot.pipeline_runs_in_period)
        )
        runs_table.add_row(
            f"Errors ({period_label})", str(snapshot.error_runs_in_period)
        )
        last_run_str = (
            snapshot.last_run_at.strftime("%Y-%m-%d %H:%M UTC")
            if snapshot.last_run_at
            else "\u2014"
        )
        runs_table.add_row("Last run at", last_run_str)
        runs_table.add_row("Parsed content rows", str(snapshot.parsed_content_total))
        console.print(runs_table)

    def save_html(self, snapshot: PipelineSnapshot, path: Optional[str] = None) -> str:
        """Generate an HTML report with a status chart and save it to disk.

        Args:
            snapshot: PipelineSnapshot containing current metrics.
            path: Output file path. Defaults to reports/monitor_YYYYMMDD_HHMM.html.

        Returns:
            Absolute path of the saved HTML file.
        """
        if path is None:
            os.makedirs("reports", exist_ok=True)
            date_str = snapshot.collected_at.strftime("%Y%m%d_%H%M")
            path = os.path.join("reports", f"monitor_{date_str}.html")

        img_b64 = self._build_chart_b64(snapshot)
        html = self._render_html(snapshot, img_b64)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

        return os.path.abspath(path)

    def _build_chart_b64(self, snapshot: PipelineSnapshot) -> str:
        """Render a status bar chart and return it as a base64-encoded PNG.

        Args:
            snapshot: PipelineSnapshot with status counts.

        Returns:
            Base64-encoded PNG string.
        """
        import matplotlib.pyplot as plt

        statuses = _STATUS_ORDER
        counts = [snapshot.search_results_by_status.get(s, 0) for s in statuses]
        colors = ["#e74c3c" if s == "failed" else "#3498db" for s in statuses]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(statuses, counts, color=colors)
        ax.set_title("Search Results by Status")
        ax.set_ylabel("Count")
        ax.set_xlabel("Status")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _render_html(self, snapshot: PipelineSnapshot, img_b64: str) -> str:
        """Build the full HTML report string.

        Args:
            snapshot: PipelineSnapshot with all metrics.
            img_b64: Base64-encoded PNG chart image.

        Returns:
            Complete HTML document as a string.
        """
        ts = snapshot.collected_at.strftime("%Y-%m-%d %H:%M UTC")
        period_label = f"last {snapshot.period_hours}h"
        last_run = (
            snapshot.last_run_at.strftime("%Y-%m-%d %H:%M UTC")
            if snapshot.last_run_at
            else "\u2014"
        )

        status_rows: list = []
        for s in _STATUS_ORDER:
            cls = "failed" if s == "failed" else ""
            count = snapshot.search_results_by_status.get(s, 0)
            status_rows.append(f'  <tr><td class="{cls}">{s}</td><td>{count}</td></tr>')
        rows_html = "\n".join(status_rows)

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            "  <title>Pipeline Monitor</title>\n"
            "  <style>\n"
            "    body { font-family: sans-serif; margin: 2rem; }\n"
            "    table { border-collapse: collapse; width: 100%;"
            " max-width: 600px; }\n"
            "    th, td { border: 1px solid #ccc; padding: 8px 12px; }\n"
            "    th { background: #f4f4f4; }\n"
            "    .failed { color: #e74c3c; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            f"  <h1>Pipeline Monitor</h1>\n"
            f"  <p>Generated: {ts}</p>\n"
            f'  <img src="data:image/png;base64,{img_b64}"'
            ' alt="Status chart" style="max-width:100%">\n'
            "  <h2>Search Results by Status</h2>\n"
            "  <table>\n"
            "    <tr><th>Status</th><th>Count</th></tr>\n"
            f"{rows_html}\n"
            "    <tr><td><strong>Total</strong></td>"
            f"<td>{snapshot.search_results_total}</td></tr>\n"
            "  </table>\n"
            "  <h2>Pipeline Runs</h2>\n"
            "  <table>\n"
            "    <tr><th>Metric</th><th>Value</th></tr>\n"
            f"    <tr><td>Total runs</td>"
            f"<td>{snapshot.pipeline_runs_total}</td></tr>\n"
            f"    <tr><td>Runs ({period_label})</td>"
            f"<td>{snapshot.pipeline_runs_in_period}</td></tr>\n"
            f"    <tr><td>Errors ({period_label})</td>"
            f"<td>{snapshot.error_runs_in_period}</td></tr>\n"
            f"    <tr><td>Last run at</td><td>{last_run}</td></tr>\n"
            f"    <tr><td>Parsed content rows</td>"
            f"<td>{snapshot.parsed_content_total}</td></tr>\n"
            "  </table>\n"
            "</body>\n"
            "</html>\n"
        )
