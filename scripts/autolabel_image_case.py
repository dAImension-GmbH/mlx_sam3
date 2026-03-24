from __future__ import annotations

import argparse
import os
from pathlib import Path

from _daimension_common import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SDK_PATH,
    MlxSam3Segmenter,
    build_sdk_client,
    download_image,
    upload_polygon_prediction,
    validate_case_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MLX SAM3 on the first image of a dAImension image case and upload polygon labels."
    )
    parser.add_argument("--dataset-id", type=int, required=True, help="dAImension dataset ID")
    parser.add_argument("--case-number", type=int, required=True, help="dAImension case number")
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
        default="Auto-generated polygon label via MLX SAM3",
        help="Description stored with each uploaded label",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="HTTP timeout for downloading source images",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.token.strip():
        raise SystemExit("A dAImension token is required. Pass --token or set DAIMENSION_TOKEN.")

    class_label = args.label_name or args.label_text
    sdk = build_sdk_client(token=args.token, sdk_path=args.sdk_path)
    case_data = sdk.get_case_by_number(dataset_id=args.dataset_id, case_number=args.case_number)
    case_id = validate_case_dataset(case_data, dataset_id=args.dataset_id, case_number=args.case_number)

    images = list(sdk.get_images_by_case(case_id))
    if not images:
        raise SystemExit(f"Case number {args.case_number} does not contain any images.")

    first_image = images[0]
    if len(images) > 1:
        print(
            "Warning: this case has multiple images. The current image-label API does not carry "
            "image_sample_id, so the uploaded label is only unambiguous for single-image cases."
        )

    image = download_image(
        str(getattr(first_image, "url")),
        timeout_seconds=args.request_timeout_seconds,
    )
    segmenter = MlxSam3Segmenter(confidence_threshold=args.confidence_threshold)
    predictions = segmenter.segment(image=image, label_text=args.label_text)
    if not predictions:
        print(
            f"No predictions above {args.confidence_threshold} were found for "
            f"sample {getattr(first_image, 'id')} ({getattr(first_image, 'name')})."
        )
        return 0

    print(
        f"Uploading {len(predictions)} polygon label(s) for first image "
        f"{getattr(first_image, 'id')} ({getattr(first_image, 'name')}) "
        f"with prompt={args.label_text!r} label={class_label!r}."
    )
    for index, prediction in enumerate(predictions, start=1):
        label_id = upload_polygon_prediction(
            sdk,
            case_id=case_id,
            class_label=class_label,
            description=args.description,
            prediction=prediction,
        )
        print(
            f"[{index}/{len(predictions)}] label_id={label_id} "
            f"score={prediction.score if prediction.score is not None else 'n/a'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
