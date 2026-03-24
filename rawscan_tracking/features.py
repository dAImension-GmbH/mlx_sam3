from __future__ import annotations

import colorsys
import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .labels import parse_polygon_points, primary_semantic_label, split_class_labels
from .types import DetectionFeatures


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    signed_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        cross = (x1 * y2) - (x2 * y1)
        signed_area += cross
        centroid_x += (x1 + x2) * cross
        centroid_y += (y1 + y2) * cross
    if abs(signed_area) < 1e-6:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    signed_area *= 0.5
    factor = 1.0 / (6.0 * signed_area)
    return (centroid_x * factor, centroid_y * factor)


def polygon_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def rasterize_polygon(points: list[tuple[float, float]], image_size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(points, fill=255, outline=255)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    inter_min_x = max(left[0], right[0])
    inter_min_y = max(left[1], right[1])
    inter_max_x = min(left[2], right[2])
    inter_max_y = min(left[3], right[3])
    inter_width = max(0.0, inter_max_x - inter_min_x)
    inter_height = max(0.0, inter_max_y - inter_min_y)
    intersection = inter_width * inter_height
    if intersection <= 0.0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    if intersection <= 0:
        return 0.0
    union = np.logical_or(left, right).sum()
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _masked_hsv_histogram(image: Image.Image, mask: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    min_x = max(0, int(math.floor(bbox[0])))
    min_y = max(0, int(math.floor(bbox[1])))
    max_x = min(image.size[0], int(math.ceil(bbox[2])))
    max_y = min(image.size[1], int(math.ceil(bbox[3])))
    if min_x >= max_x or min_y >= max_y:
        return np.zeros(24, dtype=np.float32)

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    crop = image_array[min_y:max_y, min_x:max_x]
    crop_mask = mask[min_y:max_y, min_x:max_x]
    if crop.size == 0 or not np.any(crop_mask):
        return np.zeros(24, dtype=np.float32)

    pixels = crop[crop_mask]
    hsv_pixels = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in pixels], dtype=np.float32)
    histograms: list[np.ndarray] = []
    for channel in range(3):
        histogram, _ = np.histogram(hsv_pixels[:, channel], bins=8, range=(0.0, 1.0))
        histograms.append(histogram.astype(np.float32))
    histogram_descriptor = np.concatenate(histograms)
    histogram_total = float(histogram_descriptor.sum())
    if histogram_total > 0.0:
        histogram_descriptor /= histogram_total

    mean_rgb = pixels.mean(axis=0).astype(np.float32)
    mean_hsv = hsv_pixels.mean(axis=0).astype(np.float32)
    descriptor = np.concatenate([histogram_descriptor, mean_rgb, mean_hsv]).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    if norm > 1e-6:
        descriptor /= norm
    return descriptor


def polygon_shape_signature(points: list[tuple[float, float]], centroid: tuple[float, float]) -> np.ndarray:
    distances = np.array(
        [math.dist((float(x), float(y)), centroid) for x, y in points],
        dtype=np.float32,
    )
    if distances.size == 0:
        return np.zeros(10, dtype=np.float32)
    max_distance = float(distances.max())
    if max_distance <= 1e-6:
        return np.zeros(10, dtype=np.float32)
    normalized = distances / max_distance
    histogram, _ = np.histogram(normalized, bins=8, range=(0.0, 1.0))
    area = polygon_area(points)
    perimeter = 0.0
    for index, point in enumerate(points):
        perimeter += math.dist(point, points[(index + 1) % len(points)])
    compactness = 0.0 if perimeter <= 1e-6 else float((4.0 * math.pi * area) / (perimeter * perimeter))
    descriptor = np.concatenate(
        [
            histogram.astype(np.float32),
            np.array([len(points) / 32.0, compactness], dtype=np.float32),
        ]
    )
    total = float(histogram.sum())
    if total > 0.0:
        descriptor[:8] /= total
    return descriptor


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-6:
        return 0.0
    return float(np.dot(left, right) / denominator)


def build_detection_features(
    *,
    label: Any,
    case_id: int,
    frame_group_id: int,
    frame_index: int,
    image: Image.Image,
    track_prefix: str,
) -> DetectionFeatures:
    points = parse_polygon_points(getattr(label, "label"))
    semantic_labels, tracking_tag = split_class_labels(getattr(label, "class_labels"), prefix=track_prefix)
    centroid = polygon_centroid(points)
    area = polygon_area(points)
    bbox = polygon_bbox(points)
    mask = rasterize_polygon(points, image.size)
    appearance_vector = _masked_hsv_histogram(image, mask, bbox)
    shape_vector = polygon_shape_signature(points, centroid)
    return DetectionFeatures(
        label_id=int(getattr(label, "label_id")),
        case_id=case_id,
        frame_group_id=frame_group_id,
        frame_index=frame_index,
        label=str(getattr(label, "label")),
        label_type=str(getattr(label, "label_type")),
        class_labels=list(getattr(label, "class_labels")),
        semantic_labels=semantic_labels,
        tracking_tag=tracking_tag,
        semantic_key=primary_semantic_label(getattr(label, "class_labels"), prefix=track_prefix),
        description=str(getattr(label, "description", "")),
        posted_by_id=str(getattr(label, "posted_by_id", "")),
        is_auto=bool(getattr(label, "is_auto", False)),
        points=points,
        centroid=centroid,
        area=area,
        bbox=bbox,
        mask=mask,
        appearance_vector=appearance_vector,
        shape_vector=shape_vector,
    )
