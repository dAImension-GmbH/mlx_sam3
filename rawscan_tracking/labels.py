from __future__ import annotations

import json
from collections.abc import Iterable


def normalize_class_labels(class_labels: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_label in class_labels:
        label = str(raw_label).strip()
        if not label:
            continue
        lowered = label.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(label)
    return normalized


def is_track_tag(value: str, prefix: str = "track:") -> bool:
    return str(value).strip().lower().startswith(prefix.lower())


def extract_tracking_tag(class_labels: Iterable[str], prefix: str = "track:") -> str | None:
    for label in class_labels:
        if is_track_tag(label, prefix=prefix):
            return str(label).strip()
    return None


def split_class_labels(class_labels: Iterable[str], prefix: str = "track:") -> tuple[list[str], str | None]:
    semantic_labels: list[str] = []
    tracking_tag: str | None = None
    for label in normalize_class_labels(class_labels):
        if is_track_tag(label, prefix=prefix):
            tracking_tag = label
            continue
        semantic_labels.append(label)
    return semantic_labels, tracking_tag


def build_tracking_tag(track_id: int, prefix: str = "track:") -> str:
    return f"{prefix}{int(track_id)}"


def build_updated_class_labels(
    semantic_labels: Iterable[str],
    track_id: int,
    *,
    prefix: str = "track:",
) -> list[str]:
    labels = normalize_class_labels(semantic_labels)
    labels.append(build_tracking_tag(track_id, prefix=prefix))
    return labels


def prepare_tracking_update(
    class_labels: Iterable[str],
    track_id: int,
    *,
    prefix: str = "track:",
    overwrite_existing: bool = False,
) -> tuple[list[str] | None, str | None, str]:
    semantic_labels, existing_tracking_tag = split_class_labels(class_labels, prefix=prefix)
    if existing_tracking_tag and not overwrite_existing:
        return None, existing_tracking_tag, "reserved tracking tag already present"
    return build_updated_class_labels(semantic_labels, track_id, prefix=prefix), existing_tracking_tag, ""


def primary_semantic_label(class_labels: Iterable[str], prefix: str = "track:") -> str:
    semantic_labels, _ = split_class_labels(class_labels, prefix=prefix)
    if semantic_labels:
        return semantic_labels[0]
    return "__unlabeled__"


def parse_polygon_points(label: str) -> list[tuple[float, float]]:
    payload = json.loads(label)
    geometry = payload.get("geometry") or {}
    points = geometry.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Polygon label is missing geometry.points.")

    parsed_points: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("Polygon point must contain at least two values.")
        parsed_points.append((float(point[0]), float(point[1])))
    if len(parsed_points) < 3:
        raise ValueError("A polygon needs at least three points.")
    return parsed_points
