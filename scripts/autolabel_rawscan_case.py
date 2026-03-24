from __future__ import annotations

import argparse
import os
from pathlib import Path

from _daimension_common import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SDK_PATH,
    MlxSam3Segmenter,
    RawscanTarget,
    build_sdk_client,
    download_image,
    pick_rawscan_rgb_image,
    upload_polygon_prediction,
    validate_case_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MLX SAM3 on rawscan RGB frames from a dAImension case and upload polygon labels "
            "to the selected frame groups."
        )
    )
    parser.add_argument("--dataset-id", type=int, required=True, help="dAImension dataset ID")
    parser.add_argument("--case-number", type=int, required=True, help="dAImension rawscan case number")
    parser.add_argument("--label-text", required=True, help="Text prompt to use for SAM3")
    parser.add_argument(
        "--label-name",
        default=None,
        help="Optional class label to upload instead of the prompt text",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        required=True,
        help="Minimum SAM3 confidence score to keep",
    )
    parser.add_argument("--token", default=os.environ.get("DAIMENSION_TOKEN", ""), help="dAImension API token")
    parser.add_argument(
        "--sdk-path",
        type=Path,
        default=Path(os.environ.get("DAIMENSION_SDK_PATH", DEFAULT_SDK_PATH)),
        help="Path to the dAImension Python SDK",
    )
    parser.add_argument(
        "--description",
        default="Auto-generated rawscan polygon label via MLX SAM3",
        help="Description stored with each uploaded label",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="HTTP timeout for downloading source images",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--frame-group-id",
        dest="frame_group_ids",
        action="append",
        type=int,
        help="Specific rawscan frame_group_id to process. Repeat the flag to select multiple frames.",
    )
    selection_group.add_argument(
        "--frame-index",
        dest="frame_indices",
        action="append",
        type=int,
        help="Specific rawscan frame_index to process. Repeat the flag to select multiple frames.",
    )
    return parser


def resolve_targets(rawscan_case: object, frame_group_ids: list[int] | None, frame_indices: list[int] | None) -> list[RawscanTarget]:
    frame_groups = list(getattr(rawscan_case, "frame_groups", []) or [])
    if not frame_groups:
        raise SystemExit("The rawscan case does not contain any frame groups.")

    available_group_ids = sorted(int(getattr(frame_group, "frame_group_id")) for frame_group in frame_groups)
    available_frame_indices = sorted(int(getattr(frame_group, "frame_index")) for frame_group in frame_groups)
    selected_frame_groups = frame_groups
    if frame_group_ids:
        requested_ids = {int(value) for value in frame_group_ids}
        selected_frame_groups = [
            frame_group
            for frame_group in frame_groups
            if int(getattr(frame_group, "frame_group_id")) in requested_ids
        ]
        found_ids = {int(getattr(frame_group, "frame_group_id")) for frame_group in selected_frame_groups}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            fallback_frame_groups = [
                frame_group
                for frame_group in frame_groups
                if int(getattr(frame_group, "frame_index")) in requested_ids
            ]
            fallback_indices = {int(getattr(frame_group, "frame_index")) for frame_group in fallback_frame_groups}
            if fallback_frame_groups and fallback_indices == requested_ids:
                print(
                    "Note: the values passed via --frame-group-id matched rawscan frame_index values, "
                    "not actual frame_group_id values. Treating them as frame indices."
                )
                selected_frame_groups = fallback_frame_groups
            else:
                raise SystemExit(
                    "Unknown frame_group_id values requested: "
                    f"{missing_ids}. Available frame_group_id values: {available_group_ids}. "
                    f"Available frame_index values: {available_frame_indices}. "
                    "If you meant frame positions, use --frame-index."
                )
    elif frame_indices:
        requested_indices = {int(value) for value in frame_indices}
        selected_frame_groups = [
            frame_group
            for frame_group in frame_groups
            if int(getattr(frame_group, "frame_index")) in requested_indices
        ]
        found_indices = {int(getattr(frame_group, "frame_index")) for frame_group in selected_frame_groups}
        missing_indices = sorted(requested_indices - found_indices)
        if missing_indices:
            raise SystemExit(
                "Unknown frame_index values requested: "
                f"{missing_indices}. Available frame_index values: {available_frame_indices}. "
                f"Available frame_group_id values: {available_group_ids}."
            )

    targets: list[RawscanTarget] = []
    for frame_group in selected_frame_groups:
        frame_image = pick_rawscan_rgb_image(frame_group)
        image_sample = getattr(frame_image, "image")
        targets.append(
            RawscanTarget(
                frame_group_id=int(getattr(frame_group, "frame_group_id")),
                frame_index=int(getattr(frame_group, "frame_index")),
                sample_id=int(getattr(image_sample, "id")),
                image_name=str(getattr(image_sample, "name")),
                source_url=str(getattr(image_sample, "url")),
            )
        )
    return targets


def main() -> int:
    args = build_parser().parse_args()
    if not args.token.strip():
        raise SystemExit("A dAImension token is required. Pass --token or set DAIMENSION_TOKEN.")

    class_label = args.label_name or args.label_text
    sdk = build_sdk_client(token=args.token, sdk_path=args.sdk_path)
    case_data = sdk.get_case_by_number(dataset_id=args.dataset_id, case_number=args.case_number)
    case_id = validate_case_dataset(case_data, dataset_id=args.dataset_id, case_number=args.case_number)

    rawscan_case = sdk.get_rawscan_case(case_id)
    targets = resolve_targets(rawscan_case, args.frame_group_ids, args.frame_indices)
    segmenter = MlxSam3Segmenter(confidence_threshold=args.confidence_threshold)

    total_uploaded = 0
    skipped_frames = 0
    skipped_predictions = 0
    for target_index, target in enumerate(targets, start=1):
        print(
            f"[{target_index}/{len(targets)}] frame_group_id={target.frame_group_id} "
            f"frame_index={target.frame_index} sample_id={target.sample_id} image={target.image_name} "
            f"prompt={args.label_text!r} label={class_label!r}"
        )
        try:
            image = download_image(target.source_url, timeout_seconds=args.request_timeout_seconds)
            predictions = segmenter.segment(image=image, label_text=args.label_text)
        except Exception as error:
            skipped_frames += 1
            print(f"  skipping frame_group_id={target.frame_group_id} after frame error: {error}")
            continue

        if not predictions:
            print(
                f"  no predictions above {args.confidence_threshold} for frame_group_id={target.frame_group_id}"
            )
            continue

        for prediction_index, prediction in enumerate(predictions, start=1):
            try:
                label_id = upload_polygon_prediction(
                    sdk,
                    case_id=case_id,
                    class_label=class_label,
                    description=args.description,
                    prediction=prediction,
                    frame_group_id=target.frame_group_id,
                )
            except Exception as error:
                skipped_predictions += 1
                print(
                    f"  [{prediction_index}/{len(predictions)}] skipping prediction after upload retries: {error}"
                )
                continue

            total_uploaded += 1
            print(
                f"  [{prediction_index}/{len(predictions)}] uploaded label_id={label_id} "
                f"score={prediction.score if prediction.score is not None else 'n/a'}"
            )

    print(
        f"Finished. Uploaded {total_uploaded} polygon label(s) across {len(targets)} frame group(s). "
        f"Skipped {skipped_frames} frame(s) due to frame errors and {skipped_predictions} prediction(s) due to upload errors."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
