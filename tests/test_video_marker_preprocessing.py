"""Regression tests for mapping HIVAU video markers to multimodal tokens."""
import unittest
from types import SimpleNamespace

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.train.train import preprocess_multimodal


class VideoMarkerPreprocessingTests(unittest.TestCase):
    def test_video_marker_becomes_single_image_placeholder(self):
        conversations = [[
            {"from": "human", "value": "<video>\nDescribe this clip."},
            {"from": "gpt", "value": "A person walks."},
        ]]
        result = preprocess_multimodal(
            conversations,
            SimpleNamespace(is_multimodal=True, mm_use_im_start_end=False),
            msg="Frame timing information.",
        )

        human_text = result[0][0]["value"]
        self.assertEqual(human_text.count(DEFAULT_IMAGE_TOKEN), 1)
        self.assertNotIn("<video>", human_text)
        self.assertIn("Frame timing information.", human_text)


if __name__ == "__main__":
    unittest.main()
