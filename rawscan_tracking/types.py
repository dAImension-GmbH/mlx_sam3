from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(slots=True)
class DetectionFeatures:
    label_id: int
    case_id: int
    frame_group_id: int
    frame_index: int
    label: str
    label_type: str
    class_labels: list[str]
    semantic_labels: list[str]
    tracking_tag: str | None
    semantic_key: str
    description: str
    posted_by_id: str
    is_auto: bool
    points: list[tuple[float, float]]
    centroid: tuple[float, float]
    area: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray
    appearance_vector: np.ndarray
    shape_vector: np.ndarray


@dataclass(slots=True)
class FrameContext:
    frame_group_id: int
    frame_index: int
    image_name: str
    image_url: str
    image: Image.Image
    detections: list[DetectionFeatures] = field(default_factory=list)


@dataclass(slots=True)
class CaseContext:
    dataset_id: int
    case_id: int
    case_number: int
    track_prefix: str
    class_filter: str | None
    frames: list[FrameContext]


@dataclass(slots=True)
class TrackState:
    track_id: int
    semantic_key: str
    last_detection: DetectionFeatures
    last_frame_index: int
    appearance_anchor: np.ndarray
    shape_anchor: np.ndarray
    observation_count: int = 1
    history: list[int] = field(default_factory=list)


@dataclass(slots=True)
class TrackAssignment:
    label_id: int
    frame_group_id: int
    frame_index: int
    track_id: int
    tracking_tag: str
    semantic_key: str
    score: float | None
    backend_name: str
    refinement_used: bool = False
    action: str = "pending"
    reason: str = ""
    existing_tracking_tag: str | None = None
    class_labels: list[str] = field(default_factory=list)


class TrackerBackend(Protocol):
    name: str

    def assign_tracks(self, context: CaseContext) -> list[TrackAssignment]:
        ...
