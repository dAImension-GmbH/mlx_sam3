from __future__ import annotations

from io import BytesIO
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image

from .features import build_detection_features
from .labels import primary_semantic_label
from .types import CaseContext, FrameContext

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60


def _rawscan_image_priority(frame_image: Any) -> tuple[int, int]:
    modality = int(getattr(frame_image, "modality", 0))
    image = getattr(frame_image, "image")
    image_name = str(getattr(image, "name", ""))
    rgb_priority = 0 if modality == 1 else 1
    extension_priority = 0 if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) else 1
    return (rgb_priority, extension_priority)


def pick_rawscan_rgb_image(frame_group: Any) -> Any:
    images = list(getattr(frame_group, "images", []) or [])
    if not images:
        frame_group_id = getattr(frame_group, "frame_group_id", "unknown")
        raise RuntimeError(f"Rawscan frame group {frame_group_id} does not contain any images.")
    return sorted(images, key=_rawscan_image_priority)[0]


def download_image(url: str, *, timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS) -> Image.Image:
    request = Request(url, headers={"User-Agent": "mlx-sam3-rawscan-tracking/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        image_bytes = response.read()
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def build_case_context(
    *,
    sdk: Any,
    dataset_id: int,
    case_id: int,
    case_number: int,
    rawscan_case: Any,
    track_prefix: str,
    class_filter: str | None,
    frame_group_ids: set[int] | None,
    frame_index_start: int | None,
    frame_index_end: int | None,
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> CaseContext:
    image_labels, _, _ = sdk.get_labels_of_image_case(case_id)
    image_labels = list(image_labels or [])
    labels_by_frame_group: dict[int, list[Any]] = {}
    for label in image_labels:
        frame_group_id = getattr(label, "frame_group_id", None)
        label_type = str(getattr(label, "label_type", "")).strip().lower()
        if frame_group_id is None or int(frame_group_id) <= 0 or label_type != "polygon":
            continue
        semantic_label = primary_semantic_label(getattr(label, "class_labels"), prefix=track_prefix)
        if class_filter and semantic_label != class_filter:
            continue
        labels_by_frame_group.setdefault(int(frame_group_id), []).append(label)

    frames: list[FrameContext] = []
    rawscan_frame_groups = sorted(
        list(getattr(rawscan_case, "frame_groups", []) or []),
        key=lambda frame_group: int(getattr(frame_group, "frame_index")),
    )
    for frame_group in rawscan_frame_groups:
        frame_group_id = int(getattr(frame_group, "frame_group_id"))
        frame_index = int(getattr(frame_group, "frame_index"))
        if frame_group_ids and frame_group_id not in frame_group_ids:
            continue
        if frame_index_start is not None and frame_index < frame_index_start:
            continue
        if frame_index_end is not None and frame_index > frame_index_end:
            continue

        selected_image = pick_rawscan_rgb_image(frame_group)
        image_sample = getattr(selected_image, "image")
        image = download_image(str(getattr(image_sample, "url")), timeout_seconds=timeout_seconds)

        detections = [
            build_detection_features(
                label=label,
                case_id=case_id,
                frame_group_id=frame_group_id,
                frame_index=frame_index,
                image=image,
                track_prefix=track_prefix,
            )
            for label in labels_by_frame_group.get(frame_group_id, [])
        ]

        frames.append(
            FrameContext(
                frame_group_id=frame_group_id,
                frame_index=frame_index,
                image_name=str(getattr(image_sample, "name")),
                image_url=str(getattr(image_sample, "url")),
                image=image,
                detections=detections,
            )
        )

    if not frames:
        raise RuntimeError("No rawscan frames matched the requested selection.")

    return CaseContext(
        dataset_id=dataset_id,
        case_id=case_id,
        case_number=case_number,
        track_prefix=track_prefix,
        class_filter=class_filter,
        frames=frames,
    )
