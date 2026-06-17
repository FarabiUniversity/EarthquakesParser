"""Shared helpers for KNDC bulletin listing ingestion.

This module factors out common logic used by:
- `kndc_bulletin_parser.py` (hourly incremental snapshots)
- `kndc_bulletin_populate.py` (historical backfill)

The KNDC endpoint returns JSON wrapped in a minimal HTML document.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, cast

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

LIST_URL = "https://kndc.kz/kndc/pagecontent/alarm-bulletin/getOriginList.php"


def create_session() -> requests.Session:
    """Create a requests session suitable for KNDC bulletin listing calls."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json, */*"})
    return session


def parse_html_wrapped_json(payload: str) -> object:
    """Parse KNDC's HTML-wrapped JSON response into a Python object."""
    soup = BeautifulSoup(payload or "", "html.parser")
    body = soup.body.get_text(strip=True) if soup.body else soup.get_text(strip=True)
    if not body:
        return []
    return json.loads(body)


def _extract_items(data: object) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [cast(Dict[str, Any], row) for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        items_obj: object = data.get("items", [])
        if isinstance(items_obj, list):
            return [
                cast(Dict[str, Any], row) for row in items_obj if isinstance(row, dict)
            ]
    return []


def fetch_listing(
    session: requests.Session,
    *,
    limit: int,
    timeout: int,
    desc: bool,
    start: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch a page of bulletin listing rows."""
    params: Dict[str, str | int] = {
        "orderby": "epochtime",
        "desc": "yes" if desc else "no",
        "start": int(start),
        "limit": int(limit),
    }
    try:
        resp = session.get(LIST_URL, params=params, timeout=int(timeout))
        resp.raise_for_status()
        data = parse_html_wrapped_json(resp.text)
        return _extract_items(data)
    except Exception:
        logger.exception("Failed to fetch bulletin listing")
        return []


def normalize_event(event: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Normalize a raw bulletin row and generate enriched text."""
    depth_km = float(event.get("depth", 0.0))
    parsed: Dict[str, Any] = {
        "event_id": int(event.get("id", 0)),
        "epoch_time": int(event.get("epochtime", 0)),
        "occurred_at_utc": None,
        "latitude": float(event.get("lat", 0.0)),
        "longitude": float(event.get("lon", 0.0)),
        "depth_km": depth_km,
        "mb": float(event.get("mb", 0.0)) if event.get("mb") is not None else None,
        "mpv": float(event.get("mpv", 0.0)) if event.get("mpv") is not None else None,
        "energy_class_k": float(event.get("class", 0.0))
        if event.get("class") is not None
        else None,
        "geographic_region": event.get("gregion"),
        "seismic_region": event.get("sregion"),
        "quality": event.get("qual"),
        "author": event.get("auth"),
        "last_updated": event.get("lddate"),
        "raw": event,
    }

    try:
        evdate = event.get("evdate")
        evtime = event.get("evtime")
        if evdate and evtime:
            parsed["occurred_at_utc"] = datetime.strptime(
                f"{evdate} {evtime}", "%Y-%m-%d %H:%M:%S"
            )
    except Exception:
        parsed["occurred_at_utc"] = None

    if depth_km < 70:
        depth_description = "shallow earthquake"
    elif depth_km < 300:
        depth_description = "intermediate-depth earthquake"
    else:
        depth_description = "deep earthquake"

    enriched_text = (
        f"""Official earthquake bulletin from KNDC. Event ID: {parsed['event_id']}, """
        f"""Date and time (UTC): {parsed['occurred_at_utc']}, """
        f"""Geographic region: {parsed['geographic_region']}, """
        f"""Seismic region: {parsed['seismic_region']}. """
        f"""Coordinates: Latitude {parsed['latitude']}, """
        f"""Longitude {parsed['longitude']}, """
        f"""Depth: {parsed['depth_km']} km. This was a {depth_description}. """
        f"""Magnitude scales: mb = {parsed['mb']}, mpv = {parsed['mpv']}. """
        f"""Energy class K: {parsed['energy_class_k']}. """
        f"""Source author: {parsed['author']}. """
        f"""Quality classification: {parsed['quality']}. """
        f"""Last official update: {parsed['last_updated']}"""
    ).strip()

    return parsed, enriched_text
