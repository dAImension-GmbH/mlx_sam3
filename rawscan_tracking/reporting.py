from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from .types import CaseContext, TrackAssignment


def build_report(
    *,
    context: CaseContext,
    assignments: list[TrackAssignment],
    backend_name: str,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "dry_run": dry_run,
        "track_prefix": context.track_prefix,
        "case": {
            "dataset_id": context.dataset_id,
            "case_id": context.case_id,
            "case_number": context.case_number,
            "class_filter": context.class_filter,
        },
        "frames": [
            {
                "frame_group_id": frame.frame_group_id,
                "frame_index": frame.frame_index,
                "image_name": frame.image_name,
                "detection_count": len(frame.detections),
            }
            for frame in context.frames
        ],
        "assignments": [asdict(assignment) for assignment in assignments],
    }
