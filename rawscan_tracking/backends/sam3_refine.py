from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from ..features import bbox_iou, mask_iou, polygon_bbox, rasterize_polygon
from ..types import CaseContext, DetectionFeatures, TrackAssignment, TrackState
from .local import LocalTrackerBackend


@dataclass(slots=True)
class SAM3Refiner:
    ambiguity_margin: float = 0.08
    weak_score_threshold: float = 0.22
    crop_padding_ratio: float = 0.15
    confidence_threshold: float = 0.35

    def __post_init__(self) -> None:
        self._processor: Sam3Processor | None = None

    def _processor_instance(self) -> Sam3Processor:
        if self._processor is None:
            model = build_sam3_image_model()
            self._processor = Sam3Processor(model, confidence_threshold=self.confidence_threshold)
        return self._processor

    def should_refine(self, scores: Iterable[float], *, min_score: float) -> bool:
        sorted_scores = sorted((float(score) for score in scores if score >= 0.0), reverse=True)
        if not sorted_scores:
            return False
        if sorted_scores[0] < max(min_score, self.weak_score_threshold):
            return True
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < self.ambiguity_margin:
            return True
        return False

    def _crop_from_bbox(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[int, int]]:
        width, height = image.size
        box_width = max(1.0, bbox[2] - bbox[0])
        box_height = max(1.0, bbox[3] - bbox[1])
        padding_x = box_width * self.crop_padding_ratio
        padding_y = box_height * self.crop_padding_ratio
        min_x = max(0, int(bbox[0] - padding_x))
        min_y = max(0, int(bbox[1] - padding_y))
        max_x = min(width, int(bbox[2] + padding_x))
        max_y = min(height, int(bbox[3] + padding_y))
        return image.crop((min_x, min_y, max_x, max_y)), (min_x, min_y)

    def refine_scores(
        self,
        *,
        track: TrackState,
        frame_image: Image.Image,
        candidates: list[DetectionFeatures],
        base_scores: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        semantic_prompt = track.semantic_key
        if semantic_prompt == "__unlabeled__":
            return base_scores, False

        crop, offset = self._crop_from_bbox(frame_image, track.last_detection.bbox)
        processor = self._processor_instance()
        state = processor.set_image(crop)
        output = processor.set_text_prompt(prompt=semantic_prompt, state=state)

        masks = output.get("masks")
        if masks is None:
            return base_scores, False

        masks_array = np.asarray(masks)
        if masks_array.ndim >= 4:
            mask = np.asarray(masks_array[0, 0] > 0, dtype=bool)
        elif masks_array.ndim == 3:
            mask = np.asarray(masks_array[0] > 0, dtype=bool)
        else:
            mask = np.asarray(masks_array > 0, dtype=bool)

        if mask.size == 0 or not np.any(mask):
            return base_scores, False

        ys, xs = np.where(mask)
        full_points = [
            (float(xs.min() + offset[0]), float(ys.min() + offset[1])),
            (float(xs.max() + offset[0]), float(ys.min() + offset[1])),
            (float(xs.max() + offset[0]), float(ys.max() + offset[1])),
            (float(xs.min() + offset[0]), float(ys.max() + offset[1])),
        ]
        predicted_bbox = polygon_bbox(full_points)
        predicted_mask = rasterize_polygon(full_points, frame_image.size)

        updated_scores = base_scores.copy()
        for candidate_index, candidate in enumerate(candidates):
            overlap = max(
                bbox_iou(predicted_bbox, candidate.bbox),
                mask_iou(predicted_mask, candidate.mask),
            )
            updated_scores[candidate_index] = max(updated_scores[candidate_index], (0.65 * float(updated_scores[candidate_index])) + (0.35 * overlap))
        return updated_scores, True


class SAM3RefinementTrackerBackend(LocalTrackerBackend):
    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("refiner", SAM3Refiner())
        super().__init__(**kwargs)
        self.name = "sam3-refine"

    def assign_tracks(self, context: CaseContext) -> list[TrackAssignment]:
        return super().assign_tracks(context)
