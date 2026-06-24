"""CLI entry point for the EarthquakesParser pipeline monitor."""

import argparse
import sys

from dotenv import load_dotenv


def parse_period(period_str: str) -> int:
    """Parse a period string like '24h' or '7d' into hours."""
    period_str = period_str.strip().lower()
    if period_str.endswith("d"):
        return int(period_str[:-1]) * 24
    if period_str.endswith("h"):
        return int(period_str[:-1])
    raise ValueError(f"Unknown period format: {period_str!r}. Use e.g. '24h' or '7d'.")


def main() -> None:
    """Run the pipeline monitor CLI."""
    parser = argparse.ArgumentParser(description="EarthquakesParser Pipeline Monitor")
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate an HTML report in reports/",
    )
    parser.add_argument(
        "--period",
        default="24h",
        help="Time window for pipeline run stats (e.g. 24h, 7d). Default: 24h",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Custom output path for the --html report.",
    )
    args = parser.parse_args()

    try:
        period_hours = parse_period(args.period)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    load_dotenv()

    from earthquakes_parser.monitoring.collector import MetricsCollector
    from earthquakes_parser.monitoring.reporter import PipelineReporter
    from earthquakes_parser.storage.supabase import SupabaseDB

    try:
        db = SupabaseDB()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Set SUPABASE_URL and SUPABASE_KEY environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    collector = MetricsCollector(db)
    reporter = PipelineReporter()

    snapshot = collector.collect(period_hours=period_hours)
    reporter.print_table(snapshot)

    if args.html:
        out_path = reporter.save_html(snapshot, path=args.out)
        print(f"\nHTML report saved to: {out_path}")


if __name__ == "__main__":
    main()
