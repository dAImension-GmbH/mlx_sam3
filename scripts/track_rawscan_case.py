from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _daimension_common import DEFAULT_SDK_PATH, build_sdk_client, validate_case_dataset

from rawscan_tracking.backends import LocalTrackerBackend, SAM3RefinementTrackerBackend
from rawscan_tracking.labels import prepare_tracking_update
from rawscan_tracking.reporting import build_report
from rawscan_tracking.sdk_io import build_case_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign stable track IDs across polygon labels in a dAImension rawscan case."
    )
    parser.add_argument("--dataset-id", type=int, required=True, help="dAImension dataset ID")
    parser.add_argument("--case-number", type=int, required=True, help="dAImension rawscan case number")
    parser.add_argument("--token", default=os.environ.get("DAIMENSION_TOKEN", ""), help="dAImension API token")
    parser.add_argument(
        "--sdk-path",
        type=Path,
        default=Path(os.environ.get("DAIMENSION_SDK_PATH", DEFAULT_SDK_PATH)),
        help="Path to the dAImension Python SDK",
    )
    parser.add_argument(
        "--frame-group-id",
        dest="frame_group_ids",
        action="append",
        type=int,
        help="Specific rawscan frame_group_id to process. Repeat the flag to select multiple frame groups.",
    )
    parser.add_argument("--frame-index-start", type=int, default=None, help="Inclusive rawscan frame_index lower bound.")
    parser.add_argument("--frame-index-end", type=int, default=None, help="Inclusive rawscan frame_index upper bound.")
    parser.add_argument("--class-label", default=None, help="Only process labels whose primary semantic class matches this label.")
    parser.add_argument("--backend", choices=("local", "sam3-refine"), default="local", help="Tracking backend to use.")
    parser.add_argument("--track-prefix", default="track:", help="Reserved class-label prefix used for tracking tags.")
    parser.add_argument("--max-gap", type=int, default=1, help="Maximum number of unlabeled frame gaps to bridge.")
    parser.add_argument("--min-score", type=float, default=0.4, help="Minimum assignment score for reusing an existing track.")
    parser.add_argument("--overwrite-existing-track-tags", action="store_true", help="Replace an existing reserved tracking tag instead of skipping the label.")
    parser.add_argument("--dry-run", action="store_true", help="Compute assignments without updating dAImension labels.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional path to save a JSON report.")
    return parser


def _find_detection(context, label_id: int):
    for frame in context.frames:
        for detection in frame.detections:
            if detection.label_id == label_id:
                return detection
    raise RuntimeError(f"Could not resolve detection for label_id={label_id}.")


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_summary(assignments: Iterable[object]) -> None:
    total = 0
    updated = 0
    skipped = 0
    for assignment in assignments:
        total += 1
        if getattr(assignment, "action", "") == "updated":
            updated += 1
        if getattr(assignment, "action", "").startswith("skipped"):
            skipped += 1
    print(f"Processed {total} polygon label(s): updated={updated}, skipped={skipped}.")


def main() -> int:
    args = build_parser().parse_args()
    if not args.token.strip():
        raise SystemExit("A dAImension token is required. Pass --token or set DAIMENSION_TOKEN.")

    sdk = build_sdk_client(token=args.token, sdk_path=args.sdk_path)
    case_data = sdk.get_case_by_number(dataset_id=args.dataset_id, case_number=args.case_number)
    case_id = validate_case_dataset(case_data, dataset_id=args.dataset_id, case_number=args.case_number)
    rawscan_case = sdk.get_rawscan_case(case_id)
    frame_groups = list(getattr(rawscan_case, "frame_groups", []) or [])
    if not frame_groups:
        raise SystemExit(f"Case number {args.case_number} does not appear to be a rawscan case.")

    context = build_case_context(
        sdk=sdk,
        dataset_id=args.dataset_id,
        case_id=case_id,
        case_number=args.case_number,
        rawscan_case=rawscan_case,
        track_prefix=args.track_prefix,
        class_filter=args.class_label,
        frame_group_ids=set(args.frame_group_ids or []) or None,
        frame_index_start=args.frame_index_start,
        frame_index_end=args.frame_index_end,
    )

    if args.backend == "sam3-refine":
        backend = SAM3RefinementTrackerBackend(min_score=args.min_score, max_gap=args.max_gap)
    else:
        backend = LocalTrackerBackend(min_score=args.min_score, max_gap=args.max_gap)

    assignments = backend.assign_tracks(context)
    for assignment in assignments:
        detection = _find_detection(context, assignment.label_id)
        new_class_labels, existing_tracking_tag, reason = prepare_tracking_update(
            detection.class_labels,
            assignment.track_id,
            prefix=args.track_prefix,
            overwrite_existing=args.overwrite_existing_track_tags,
        )
        assignment.existing_tracking_tag = existing_tracking_tag
        if new_class_labels is None:
            assignment.action = "skipped-existing-track-tag"
            assignment.reason = reason
            continue

        assignment.class_labels = new_class_labels
        if args.dry_run:
            assignment.action = "dry-run"
            assignment.reason = "write skipped by --dry-run"
            continue

        successful = bool(
            sdk.update_image_label_to_case(
                case_id=case_id,
                label_id=detection.label_id,
                class_labels=new_class_labels,
                label=detection.label,
                label_type=detection.label_type,
                description=detection.description,
                is_auto=detection.is_auto,
                frame_group_id=detection.frame_group_id,
            )
        )
        assignment.action = "updated" if successful else "update-failed"
        assignment.reason = "" if successful else "SDK update_image_label_to_case returned false"

        print(
            f"frame_index={assignment.frame_index} frame_group_id={assignment.frame_group_id} "
            f"label_id={assignment.label_id} track={assignment.tracking_tag} action={assignment.action}"
        )

    report = build_report(context=context, assignments=assignments, backend_name=backend.name, dry_run=args.dry_run)
    if args.report_path is not None:
        _write_report(args.report_path, report)
        print(f"Saved report to {args.report_path}")

    _print_summary(assignments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
