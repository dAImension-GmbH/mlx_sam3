from __future__ import annotations

import unittest

from rawscan_tracking.labels import parse_polygon_points, prepare_tracking_update, split_class_labels


class RawscanTrackingLabelsTest(unittest.TestCase):
    def test_split_class_labels_preserves_semantic_labels(self) -> None:
        semantic_labels, tracking_tag = split_class_labels(["person", "track:12", "person", ""])
        self.assertEqual(semantic_labels, ["person"])
        self.assertEqual(tracking_tag, "track:12")

    def test_prepare_tracking_update_skips_existing_tag_without_overwrite(self) -> None:
        updated_labels, existing_tag, reason = prepare_tracking_update(
            ["person", "track:3"],
            7,
            overwrite_existing=False,
        )
        self.assertIsNone(updated_labels)
        self.assertEqual(existing_tag, "track:3")
        self.assertIn("already present", reason)

    def test_prepare_tracking_update_replaces_existing_tag_when_overwriting(self) -> None:
        updated_labels, existing_tag, reason = prepare_tracking_update(
            ["person", "track:3"],
            7,
            overwrite_existing=True,
        )
        self.assertEqual(updated_labels, ["person", "track:7"])
        self.assertEqual(existing_tag, "track:3")
        self.assertEqual(reason, "")

    def test_parse_polygon_points_reads_selector_json(self) -> None:
        points = parse_polygon_points(
            '{"type":"POLYGON","geometry":{"points":[[0,0],[10,0],[10,5],[0,5]]}}'
        )
        self.assertEqual(points, [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])


if __name__ == "__main__":
    unittest.main()
