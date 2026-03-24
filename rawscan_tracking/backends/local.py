from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..association import build_score_matrix, solve_assignment
from ..labels import build_tracking_tag
from ..types import CaseContext, DetectionFeatures, TrackAssignment, TrackState


@dataclass(slots=True)
class LocalTrackerBackend:
    min_score: float = 0.4
    max_gap: int = 1
    max_centroid_distance: float = 0.0
    min_area_ratio: float = 0.2
    min_overlap: float = 0.0
    refiner: object | None = None
    name: str = "local"

    @staticmethod
    def _update_track_anchor(anchor: np.ndarray, value: np.ndarray, observation_count: int) -> np.ndarray:
        blended = ((anchor * float(observation_count)) + value) / float(observation_count + 1)
        norm = float(np.linalg.norm(blended))
        if norm > 1e-6:
            blended = blended / norm
        return blended.astype(np.float32)

    def assign_tracks(self, context: CaseContext) -> list[TrackAssignment]:
        active_tracks: dict[str, list[TrackState]] = defaultdict(list)
        assignments: list[TrackAssignment] = []
        next_track_id = 1

        for frame in context.frames:
            detections_by_class: dict[str, list[DetectionFeatures]] = defaultdict(list)
            for detection in frame.detections:
                detections_by_class[detection.semantic_key].append(detection)

            for semantic_key, detections in detections_by_class.items():
                detections = sorted(detections, key=lambda detection: detection.label_id)
                candidate_tracks = [
                    track
                    for track in active_tracks.get(semantic_key, [])
                    if 0 < (frame.frame_index - track.last_frame_index) <= (self.max_gap + 1)
                ]

                score_matrix = build_score_matrix(
                    candidate_tracks,
                    detections,
                    image_size=frame.image.size,
                    max_gap=self.max_gap,
                    max_centroid_distance=self.max_centroid_distance,
                    min_area_ratio=self.min_area_ratio,
                    min_overlap=self.min_overlap,
                )

                refinement_used = False
                if self.refiner is not None and score_matrix.size > 0:
                    for row_index, track in enumerate(candidate_tracks):
                        if self.refiner.should_refine(score_matrix[row_index], min_score=self.min_score):
                            refined_scores, used = self.refiner.refine_scores(
                                track=track,
                                frame_image=frame.image,
                                candidates=detections,
                                base_scores=score_matrix[row_index],
                            )
                            score_matrix[row_index] = refined_scores
                            refinement_used = refinement_used or used

                matches = solve_assignment(score_matrix, min_score=self.min_score)
                matched_detection_indices = {column_index for _, column_index, _ in matches}

                for row_index, column_index, score in matches:
                    track = candidate_tracks[row_index]
                    detection = detections[column_index]
                    track.appearance_anchor = self._update_track_anchor(
                        track.appearance_anchor,
                        detection.appearance_vector,
                        track.observation_count,
                    )
                    track.shape_anchor = self._update_track_anchor(
                        track.shape_anchor,
                        detection.shape_vector,
                        track.observation_count,
                    )
                    track.last_detection = detection
                    track.last_frame_index = frame.frame_index
                    track.observation_count += 1
                    track.history.append(detection.label_id)
                    assignments.append(
                        TrackAssignment(
                            label_id=detection.label_id,
                            frame_group_id=detection.frame_group_id,
                            frame_index=detection.frame_index,
                            track_id=track.track_id,
                            tracking_tag=build_tracking_tag(track.track_id, prefix=context.track_prefix),
                            semantic_key=semantic_key,
                            score=score,
                            backend_name=self.name,
                            refinement_used=refinement_used,
                            existing_tracking_tag=detection.tracking_tag,
                            class_labels=list(detection.class_labels),
                        )
                    )

                for detection_index, detection in enumerate(detections):
                    if detection_index in matched_detection_indices:
                        continue
                    track = TrackState(
                        track_id=next_track_id,
                        semantic_key=semantic_key,
                        last_detection=detection,
                        last_frame_index=frame.frame_index,
                        appearance_anchor=detection.appearance_vector.copy(),
                        shape_anchor=detection.shape_vector.copy(),
                        observation_count=1,
                        history=[detection.label_id],
                    )
                    active_tracks[semantic_key].append(track)
                    assignments.append(
                        TrackAssignment(
                            label_id=detection.label_id,
                            frame_group_id=detection.frame_group_id,
                            frame_index=detection.frame_index,
                            track_id=next_track_id,
                            tracking_tag=build_tracking_tag(next_track_id, prefix=context.track_prefix),
                            semantic_key=semantic_key,
                            score=None,
                            backend_name=self.name,
                            refinement_used=False,
                            existing_tracking_tag=detection.tracking_tag,
                            class_labels=list(detection.class_labels),
                        )
                    )
                    next_track_id += 1

        assignments.sort(key=lambda assignment: (assignment.frame_index, assignment.label_id))
        return assignments
