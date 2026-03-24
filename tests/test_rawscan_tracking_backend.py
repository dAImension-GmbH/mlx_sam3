from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from PIL import Image, ImageDraw

from rawscan_tracking.backends.local import LocalTrackerBackend
from rawscan_tracking.features import build_detection_features
from rawscan_tracking.types import CaseContext, FrameContext


@dataclass
class DummyLabel:
    label_id: int
    label: str
    label_type: str
    class_labels: list[str]
    description: str = ""
    posted_by_id: str = "tester"
    is_auto: bool = False


def polygon_label(points: list[tuple[float, float]]) -> str:
    return json.dumps({"type": "POLYGON", "geometry": {"points": points}})


def make_frame(
    *,
    frame_group_id: int,
    frame_index: int,
    shapes: list[tuple[int, tuple[int, int, int], list[tuple[float, float]]]],
) -> FrameContext:
    image = Image.new("RGB", (40, 20), "black")
    draw = ImageDraw.Draw(image)
    detections = []
    for label_id, color, points in shapes:
        draw.polygon(points, fill=color)
        detections.append(
            build_detection_features(
                label=DummyLabel(
                    label_id=label_id,
                    label=polygon_label(points),
                    label_type="Polygon",
                    class_labels=["person"],
                ),
                case_id=101,
                frame_group_id=frame_group_id,
                frame_index=frame_index,
                image=image,
                track_prefix="track:",
            )
        )
    return FrameContext(
        frame_group_id=frame_group_id,
        frame_index=frame_index,
        image_name=f"frame-{frame_index}.png",
        image_url=f"https://example.test/frame-{frame_index}.png",
        image=image,
        detections=detections,
    )


class RawscanTrackingBackendTest(unittest.TestCase):
    def test_assigns_consistent_tracks_for_crossing_same_class_objects(self) -> None:
        frame_one = make_frame(
            frame_group_id=11,
            frame_index=1,
            shapes=[
                (1, (255, 0, 0), [(2, 4), (12, 4), (12, 14), (2, 14)]),
                (2, (0, 0, 255), [(26, 4), (36, 4), (36, 14), (26, 14)]),
            ],
        )
        frame_two = make_frame(
            frame_group_id=12,
            frame_index=2,
            shapes=[
                (3, (0, 0, 255), [(4, 4), (14, 4), (14, 14), (4, 14)]),
                (4, (255, 0, 0), [(24, 4), (34, 4), (34, 14), (24, 14)]),
            ],
        )
        context = CaseContext(
            dataset_id=9,
            case_id=101,
            case_number=5,
            track_prefix="track:",
            class_filter=None,
            frames=[frame_one, frame_two],
        )

        assignments = LocalTrackerBackend(min_score=0.45, max_gap=1).assign_tracks(context)
        track_by_label = {assignment.label_id: assignment.track_id for assignment in assignments}

        self.assertEqual(track_by_label[1], 1)
        self.assertEqual(track_by_label[2], 2)
        self.assertEqual(track_by_label[4], 1)
        self.assertEqual(track_by_label[3], 2)

    def test_reuses_track_across_short_empty_gap(self) -> None:
        frame_one = make_frame(
            frame_group_id=21,
            frame_index=1,
            shapes=[(1, (0, 255, 0), [(5, 5), (13, 5), (13, 13), (5, 13)])],
        )
        empty_frame = FrameContext(
            frame_group_id=22,
            frame_index=2,
            image_name="frame-2.png",
            image_url="https://example.test/frame-2.png",
            image=Image.new("RGB", (40, 20), "black"),
            detections=[],
        )
        frame_three = make_frame(
            frame_group_id=23,
            frame_index=3,
            shapes=[(2, (0, 255, 0), [(7, 5), (15, 5), (15, 13), (7, 13)])],
        )
        context = CaseContext(
            dataset_id=9,
            case_id=101,
            case_number=5,
            track_prefix="track:",
            class_filter=None,
            frames=[frame_one, empty_frame, frame_three],
        )

        assignments = LocalTrackerBackend(min_score=0.35, max_gap=1).assign_tracks(context)
        track_by_label = {assignment.label_id: assignment.track_id for assignment in assignments}

        self.assertEqual(track_by_label[1], track_by_label[2])


if __name__ == "__main__":
    unittest.main()
