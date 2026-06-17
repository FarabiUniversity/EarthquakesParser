"""Supabase ingestion helpers for KNDC parser outputs."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from earthquakes_parser.storage.supabase.database import SupabaseDB


def _json_safe(value: Any) -> Any:
    """Convert nested values into JSON-serializable structures."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return value


def _normalize_timestamp(value: Any, *, assume_utc: bool = False) -> Optional[str]:
    """Normalize timestamps for Supabase storage."""
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None and assume_utc:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()

    text = str(value).strip()
    return text or None


def _extract_newsid(source_url: str) -> Optional[int]:
    """Extract the KNDC newsid from a source URL."""
    try:
        if "news=" not in source_url:
            return None
        return int(source_url.split("news=")[-1].split("&")[0])
    except Exception:
        return None


class KndcSupabaseStore:
    """Store KNDC parser output in Supabase using natural-key upserts."""

    NEWS_TABLE = "kndc_news"
    BULLETIN_TABLE = "kndc_bulletins"

    def __init__(self, db: Optional[SupabaseDB] = None) -> None:
        """Initialize the store with an existing or default Supabase client."""
        self.db = db or SupabaseDB()

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return float(min(2**attempt, 8))

    def _execute_with_retry(self, action, *, operation: str, attempts: int = 3):
        """Run a Supabase action with a small retry budget.

        This protects ingestion from transient transport failures such as
        ``httpx.RemoteProtocolError`` while keeping the call sites simple.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return action()
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(self._retry_delay(attempt))

        if last_error is not None:
            raise RuntimeError(
                f"{operation} failed after {attempts} attempts"
            ) from last_error
        raise RuntimeError(f"{operation} failed without raising an exception")

    def _latest_integer(self, table: str, column: str) -> Optional[int]:
        response = self._execute_with_retry(
            lambda: (
                self.db.client.table(table)
                .select(column)
                .order(column, desc=True)
                .limit(1)
                .execute()
            ),
            operation=f"read latest {column} from {table}",
        )
        rows = getattr(response, "data", None) or []
        if not rows:
            return None
        raw_value = rows[0].get(column)
        if raw_value is None:
            return None
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return None

    def latest_newsid(self) -> Optional[int]:
        """Return the highest KNDC newsid currently in Supabase."""
        return self._latest_integer(self.NEWS_TABLE, "newsid")

    def latest_event_id(self) -> Optional[int]:
        """Return the highest KNDC bulletin event id currently in Supabase."""
        return self._latest_integer(self.BULLETIN_TABLE, "event_id")

    def bulletin_exists(self, event_id: int) -> bool:
        """Return True when a bulletin row already exists for `event_id`."""
        response = self._execute_with_retry(
            lambda: (
                self.db.client.table(self.BULLETIN_TABLE)
                .select("id")
                .eq("event_id", int(event_id))
                .limit(1)
                .execute()
            ),
            operation=f"check bulletin existence for event_id={int(event_id)}",
        )
        rows = getattr(response, "data", None) or []
        return bool(rows)

    def upsert_news(self, record: Any) -> Dict[str, Any]:
        """Upsert a KNDC news record into Supabase."""
        source_url = str(getattr(record, "source_url", "")).strip()
        newsid = _extract_newsid(source_url)
        if newsid is None:
            raise ValueError(f"Could not extract newsid from source_url: {source_url}")

        payload = {
            "newsid": newsid,
            "source": str(getattr(record, "source", "kndc")),
            "source_url": source_url,
            "title": str(getattr(record, "title", "")),
            "published_at": _normalize_timestamp(getattr(record, "published_at", None)),
            "main_text": str(getattr(record, "main_text", "")),
            "raw": _json_safe(getattr(record, "raw", {})),
        }

        response = self._execute_with_retry(
            lambda: (
                self.db.client.table(self.NEWS_TABLE)
                .upsert(payload, on_conflict="newsid")
                .execute()
            ),
            operation=f"upsert newsid={newsid} into {self.NEWS_TABLE}",
        )
        rows = getattr(response, "data", None) or []
        return dict(rows[0]) if rows else payload

    def upsert_bulletin(
        self,
        event_id: int,
        parsed: Dict[str, Any],
        enriched_text: str,
    ) -> Dict[str, Any]:
        """Upsert a KNDC bulletin record into Supabase."""
        payload = {
            "event_id": int(event_id),
            "source": "kndc",
            "source_url": (
                "https://kndc.kz/kndc/pagecontent/alarm-bulletin/"
                f"getOriginList.php?event_id={int(event_id)}"
            ),
            "epoch_time": int(parsed.get("epoch_time") or 0),
            "occurred_at_utc": _normalize_timestamp(
                parsed.get("occurred_at_utc"), assume_utc=True
            ),
            "latitude": parsed.get("latitude"),
            "longitude": parsed.get("longitude"),
            "depth_km": parsed.get("depth_km"),
            "mb": parsed.get("mb"),
            "mpv": parsed.get("mpv"),
            "energy_class_k": parsed.get("energy_class_k"),
            "geographic_region": parsed.get("geographic_region"),
            "seismic_region": parsed.get("seismic_region"),
            "quality": parsed.get("quality"),
            "author": parsed.get("author"),
            "last_updated": _normalize_timestamp(parsed.get("last_updated")),
            "parsed": _json_safe(parsed),
            "enriched_text": str(enriched_text or ""),
            "raw": _json_safe(parsed.get("raw", {})),
        }

        response = self._execute_with_retry(
            lambda: (
                self.db.client.table(self.BULLETIN_TABLE)
                .upsert(payload, on_conflict="event_id")
                .execute()
            ),
            operation=f"upsert event_id={int(event_id)} into {self.BULLETIN_TABLE}",
        )
        rows = getattr(response, "data", None) or []
        return dict(rows[0]) if rows else payload
