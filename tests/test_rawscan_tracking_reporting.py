from __future__ import annotations

import unittest

from PIL import Image

from rawscan_tracking.reporting import build_report
from rawscan_tracking.types import CaseContext, FrameContext, TrackAssignment


class RawscanTrackingReportingTest(unittest.TestCase):
    def test_build_report_contains_case_frame_and_assignment_data(self) -> None:
        context = CaseContext(
            dataset_id=4,
            case_id=8,
            case_number=12,
            track_prefix="track:",
            class_filter="person",
            frames=[
                FrameContext(
                    frame_group_id=77,
                    frame_index=3,
                    image_name="frame-3.png",
                    image_url="https://example.test/frame-3.png",
                    image=Image.new("RGB", (8, 8), "black"),
                    detections=[],
                )
            ],
        )
        assignments = [
            TrackAssignment(
                label_id=99,
                frame_group_id=77,
                frame_index=3,
                track_id=2,
                tracking_tag="track:2",
                semantic_key="person",
                score=0.91,
                backend_name="local",
                action="dry-run",
                reason="write skipped by --dry-run",
                class_labels=["person", "track:2"],
            )
        ]

        report = build_report(context=context, assignments=assignments, backend_name="local", dry_run=True)

        self.assertEqual(report["backend"], "local")
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["case"]["case_number"], 12)
        self.assertEqual(report["frames"][0]["frame_group_id"], 77)
        self.assertEqual(report["assignments"][0]["tracking_tag"], "track:2")


if __name__ == "__main__":
    unittest.main()
