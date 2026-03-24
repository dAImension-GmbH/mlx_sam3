from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_SDK_PATH = Path(__file__).resolve().parents[2] / "dAImension-Python-SDK"


@dataclass(frozen=True, slots=True)
class SegmentationPrediction:
    points: list[tuple[float, float]]
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RawscanTarget:
    frame_group_id: int
    frame_index: int
    sample_id: int
    image_name: str
    source_url: str


def ensure_sdk_import_path(sdk_path: Path) -> None:
    if not sdk_path.exists():
        raise RuntimeError(f"dAImension SDK path does not exist: {sdk_path}")
    sdk_path_str = str(sdk_path)
    if sdk_path_str not in sys.path:
        sys.path.insert(0, sdk_path_str)


def build_sdk_client(token: str, sdk_path: Path) -> Any:
    try:
        from dAImension import dAImension
    except ModuleNotFoundError as error:
        missing_name = error.name or "unknown"
        if missing_name in {"dAImension", "grpc_generated_files", "models"}:
            ensure_sdk_import_path(sdk_path)
            try:
                from dAImension import dAImension
            except ModuleNotFoundError as nested_error:
                nested_name = nested_error.name or "unknown"
                if nested_name in {"google", "google.protobuf", "grpc"}:
                    raise RuntimeError(
                        "The dAImension SDK dependencies are not installed in this environment. "
                        "Run `uv sync` in `/Users/mdklause/dAimension Forge/mlx_sam3`, then rerun the script."
                    ) from nested_error
                raise
        elif missing_name in {"google", "google.protobuf", "grpc"}:
            raise RuntimeError(
                "The dAImension SDK dependencies are not installed in this environment. "
                "Run `uv sync` in `/Users/mdklause/dAimension Forge/mlx_sam3`, then rerun the script."
            ) from error
        else:
            raise

    return dAImension(token=token)


def download_image(url: str, timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS) -> Image.Image:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            request = Request(url, headers={"User-Agent": "mlx-sam3-dAImension-script/0.1"})
            with urlopen(request, timeout=timeout_seconds) as response:
                image_bytes = response.read()
            image = Image.open(BytesIO(image_bytes))
            return image.convert("RGB")
        except (IncompleteRead, OSError, URLError) as error:
            last_error = error
            if attempt == 0:
                print(f"  download failed once, retrying: {error}")
                continue
            break

    assert last_error is not None
    raise RuntimeError(f"Failed to download image after 2 attempts: {last_error}") from last_error


def build_polygon_label_json(points: Iterable[tuple[float, float]]) -> str:
    normalized_points = [[float(x), float(y)] for x, y in points]
    if len(normalized_points) < 3:
        raise ValueError("A polygon needs at least three points.")
    xs = [point[0] for point in normalized_points]
    ys = [point[1] for point in normalized_points]
    payload = {
        "type": "POLYGON",
        "geometry": {
            "bounds": {
                "minX": min(xs),
                "minY": min(ys),
                "maxX": max(xs),
                "maxY": max(ys),
            },
            "points": normalized_points,
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def _cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    sorted_points = sorted(set((float(x), float(y)) for x, y in points))
    if len(sorted_points) <= 2:
        return list(sorted_points)

    lower: list[tuple[float, float]] = []
    for point in sorted_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _box_to_polygon(box: Any) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _mask_to_polygon(mask: Any, fallback_box: Any | None = None) -> list[tuple[float, float]]:
    mask_array = np.asarray(mask)
    while mask_array.ndim > 2:
        mask_array = mask_array[0]

    positive = np.argwhere(mask_array > 0)
    if positive.size == 0:
        return _box_to_polygon(fallback_box) if fallback_box is not None else []

    max_points = int(os.environ.get("SAM3_MASK_MAX_POINTS", "4000"))
    if len(positive) > max_points:
        step = max(1, len(positive) // max_points)
        positive = positive[::step]

    xy_points = [(float(x), float(y)) for y, x in positive]
    hull = _convex_hull(xy_points)
    if len(hull) >= 3:
        return hull
    return _box_to_polygon(fallback_box) if fallback_box is not None else []


def _to_indexed_items(value: Any) -> list[Any]:
    if value is None:
        return []
    array = np.asarray(value)
    if array.ndim == 0:
        return [array.item()]
    if array.ndim == 1:
        if array.shape[0] == 4:
            return [array]
        return [array[index].item() for index in range(array.shape[0])]
    return [array[index] for index in range(array.shape[0])]


def _score_to_float(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    return float(array.reshape(-1).max())


class MlxSam3Segmenter:
    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = float(confidence_threshold)
        model = build_sam3_image_model()
        self.processor = Sam3Processor(model, confidence_threshold=self.confidence_threshold)

    def segment(self, image: Image.Image, label_text: str) -> list[SegmentationPrediction]:
        state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(prompt=label_text, state=state)

        masks = _to_indexed_items(output.get("masks"))
        boxes = _to_indexed_items(output.get("boxes"))
        scores = _to_indexed_items(output.get("scores"))

        predictions: list[SegmentationPrediction] = []
        for index, mask in enumerate(masks):
            score = _score_to_float(scores[index]) if index < len(scores) else None
            if score is not None and score < self.confidence_threshold:
                continue
            fallback_box = boxes[index] if index < len(boxes) else None
            polygon = _mask_to_polygon(mask, fallback_box=fallback_box)
            if len(polygon) < 3:
                continue
            predictions.append(SegmentationPrediction(points=polygon, score=score))

        if predictions:
            return predictions

        for index, box in enumerate(boxes):
            score = _score_to_float(scores[index]) if index < len(scores) else None
            if score is not None and score < self.confidence_threshold:
                continue
            predictions.append(SegmentationPrediction(points=_box_to_polygon(box), score=score))

        return predictions


def upload_polygon_prediction(
    sdk: Any,
    *,
    case_id: int,
    class_label: str,
    description: str,
    prediction: SegmentationPrediction,
    frame_group_id: int | None = None,
) -> int:
    _, label_id = sdk.add_image_label_to_case(
        case_id=case_id,
        class_labels=[class_label],
        label=build_polygon_label_json(prediction.points),
        label_type="POLYGON",
        description=description,
        is_auto=True,
        frame_group_id=frame_group_id,
    )
    return int(label_id)


def validate_case_dataset(case_data: Any, dataset_id: int, case_number: int) -> int:
    actual_dataset_id = int(getattr(case_data, "dataset_id"))
    actual_case_number = int(getattr(case_data, "case_number"))
    if actual_dataset_id != dataset_id:
        raise RuntimeError(
            f"Case number {case_number} belongs to dataset {actual_dataset_id}, not requested dataset {dataset_id}."
        )
    if actual_case_number != case_number:
        raise RuntimeError(
            f"Resolved case number {actual_case_number} does not match requested case number {case_number}."
        )
    return int(getattr(case_data, "id"))


def _rawscan_image_priority(frame_image: Any) -> tuple[int, str]:
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
