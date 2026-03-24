from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import bbox_iou, cosine_similarity, mask_iou
from .types import DetectionFeatures, TrackState


def area_ratio_score(left_area: float, right_area: float) -> float:
    if left_area <= 1e-6 or right_area <= 1e-6:
        return 0.0
    return min(left_area, right_area) / max(left_area, right_area)


def centroid_score(
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    image_size: tuple[int, int],
    frame_gap: int,
) -> float:
    diagonal = math.hypot(float(image_size[0]), float(image_size[1])) or 1.0
    normalized_distance = math.dist(left, right) / (diagonal * max(1, frame_gap))
    return max(0.0, 1.0 - normalized_distance * 2.0)


def score_detection_match(
    track: TrackState,
    detection: DetectionFeatures,
    *,
    image_size: tuple[int, int],
    frame_gap: int,
) -> float:
    overlap_score = max(
        mask_iou(track.last_detection.mask, detection.mask),
        bbox_iou(track.last_detection.bbox, detection.bbox),
    )
    centroid_similarity = centroid_score(
        track.last_detection.centroid,
        detection.centroid,
        image_size=image_size,
        frame_gap=frame_gap,
    )
    area_similarity = area_ratio_score(track.last_detection.area, detection.area)
    appearance_similarity = max(
        0.0,
        min(
            1.0,
            (
                0.4 * cosine_similarity(track.last_detection.appearance_vector, detection.appearance_vector)
                + 0.6 * cosine_similarity(track.appearance_anchor, detection.appearance_vector)
            ),
        ),
    )
    shape_similarity = max(
        0.0,
        min(
            1.0,
            (
                0.4 * cosine_similarity(track.last_detection.shape_vector, detection.shape_vector)
                + 0.6 * cosine_similarity(track.shape_anchor, detection.shape_vector)
            ),
        ),
    )
    additive_score = (
        (0.20 * overlap_score)
        + (0.15 * centroid_similarity)
        + (0.15 * area_similarity)
        + (0.35 * appearance_similarity)
        + (0.15 * shape_similarity)
    )
    appearance_gate = 0.2 + (0.8 * appearance_similarity)
    return additive_score * appearance_gate


def build_score_matrix(
    tracks: list[TrackState],
    detections: list[DetectionFeatures],
    *,
    image_size: tuple[int, int],
    max_gap: int,
    max_centroid_distance: float,
    min_area_ratio: float,
    min_overlap: float,
) -> np.ndarray:
    score_matrix = np.full((len(tracks), len(detections)), -1.0, dtype=np.float32)
    for row_index, track in enumerate(tracks):
        frame_gap = detections[0].frame_index - track.last_frame_index if detections else 0
        if frame_gap <= 0 or frame_gap > (max_gap + 1):
            continue
        for column_index, detection in enumerate(detections):
            bbox_overlap = bbox_iou(track.last_detection.bbox, detection.bbox)
            mask_overlap = mask_iou(track.last_detection.mask, detection.mask)
            overlap = max(bbox_overlap, mask_overlap)
            area_similarity = area_ratio_score(track.last_detection.area, detection.area)
            motion = centroid_score(
                track.last_detection.centroid,
                detection.centroid,
                image_size=image_size,
                frame_gap=frame_gap,
            )
            if motion < max_centroid_distance or area_similarity < min_area_ratio or overlap < min_overlap:
                continue
            score_matrix[row_index, column_index] = score_detection_match(
                track,
                detection,
                image_size=image_size,
                frame_gap=frame_gap,
            )
    return score_matrix


def solve_assignment(score_matrix: np.ndarray, *, min_score: float) -> list[tuple[int, int, float]]:
    if score_matrix.size == 0:
        return []

    cost_matrix = np.where(score_matrix >= 0.0, 1.0 - score_matrix, 10.0)
    row_indices, column_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int, float]] = []
    for row_index, column_index in zip(row_indices, column_indices, strict=False):
        score = float(score_matrix[row_index, column_index])
        if score < min_score:
            continue
        matches.append((int(row_index), int(column_index), score))
    return matches
