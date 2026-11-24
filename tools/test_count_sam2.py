"""CPU checks for duplicate masks, consecutive persistence and object births."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from video.count_sam2 import discover, independent_masks, persistent, track_long_term


def obj(pixels, box_id=0):
    return dict(mask=np.array(pixels, dtype=np.int32), box=np.array([box_id, 0, 1, 1]))


class PipelineTests(unittest.TestCase):
    def test_duplicate_prompts_and_empty_masks_do_not_become_distinct_objects(self):
        class ImagePredictor:
            def set_image(self, image):
                pass

            def predict(self, **kwargs):
                return (np.array([[[[1, 0], [0, 0]]], [[[1, 0], [0, 0]]], [[[0, 0], [0, 0]]]]),
                        np.array([[0.5], [0.9], [0.8]]), None)
        boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [1, 1, 2, 2]])
        kept = independent_masks(ImagePredictor(), Image.new("RGB", (2, 2)), boxes)
        self.assertEqual(len(kept), 1)
        np.testing.assert_array_equal(kept[0]["box"], boxes[1])

    def test_temporal_filter_requires_consecutive_matches(self):
        independent = [[obj([1, 2])] for _ in range(5)]
        tracked = {i: {1: obj([1, 2])["mask"]} for i in range(5)}
        self.assertTrue(persistent(0, 1, tracked, independent, 3))
        independent[1] = []
        self.assertFalse(persistent(0, 1, tracked, independent, 3))
        # An isolated later match cannot bridge the missing frame.
        self.assertTrue(persistent(3, 1, tracked, independent, 3))

    def test_new_object_must_be_uncovered_and_nonempty(self):
        births = discover([obj([]), obj([2, 3]), obj([4, 5]), obj([4, 5])], np.array([1, 2]))
        self.assertEqual(len(births), 1)
        np.testing.assert_array_equal(births[0]["mask"], [4, 5])

    def test_empty_start_late_birth_and_multiple_batches_have_unique_ids(self):
        independent = [[], [obj([1], 1)], [obj([1], 1), obj([2], 2), obj([3], 3), obj([4], 4)]]
        calls = []

        def propagate(predictor, state, frame, seeds):
            calls.append((frame, list(seeds)))
            for i in range(frame, len(independent)):
                yield i, {obj_id: np.array([int(box[0])], dtype=np.int32) for obj_id, box in seeds}

        with patch("video.count_sam2.propagate", propagate):
            tracks = track_long_term(None, None, independent, batch_size=2)
        self.assertEqual(len(tracks), 4)
        self.assertEqual(set(tracks), {1, 2, 3, 4})
        self.assertEqual([len(seeds) for _, seeds in calls], [1, 2, 1])
        self.assertTrue(all(0 not in trajectory for trajectory in tracks.values()))

    def test_no_detections_is_zero_without_seeding_sam(self):
        with patch("video.count_sam2.propagate") as propagate:
            self.assertEqual(track_long_term(None, None, [[], [], []]), {})
            propagate.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
