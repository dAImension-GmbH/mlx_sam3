from __future__ import annotations

import argparse
import os
from pathlib import Path

from _daimension_common import DEFAULT_SDK_PATH, build_sdk_client, validate_case_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete all image labels from a dAImension rawscan case."
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
    return parser


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

    image_labels, _, _ = sdk.get_labels_of_image_case(case_id)
    image_labels = list(image_labels)
    if not image_labels:
        print(f"No image labels found for rawscan case number {args.case_number}.")
        return 0

    deleted_count = 0
    for index, label in enumerate(image_labels, start=1):
        label_id = int(getattr(label, "label_id"))
        frame_group_id = getattr(label, "frame_group_id", None)
        successful = bool(sdk.remove_label_from_case(case_id=case_id, label_id=label_id))
        if successful:
            deleted_count += 1
        print(
            f"[{index}/{len(image_labels)}] label_id={label_id} "
            f"frame_group_id={frame_group_id if frame_group_id is not None else 'n/a'} "
            f"deleted={successful}"
        )

    print(
        f"Finished. Deleted {deleted_count} of {len(image_labels)} image label(s) "
        f"from rawscan case number {args.case_number}."
    )
    return 0 if deleted_count == len(image_labels) else 1


if __name__ == "__main__":
    raise SystemExit(main())
