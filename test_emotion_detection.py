import unittest
from unittest.mock import patch, MagicMock
from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    def _mock_response(self, status_code=200, emotions=None):
        """Helper to build a fake API response."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        if emotions is None:
            emotions = {"anger": 0.01, "joy": 0.95, "sadness": 0.04}
        mock_response.json.return_value = {
            "emotionPredictions": [{"emotion": emotions}]
        }
        return mock_response

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_joy(self, mock_post):
        mock_post.return_value = self._mock_response(
            emotions={"anger": 0.01, "joy": 0.95, "sadness": 0.04}
        )
        result = emotion_detector("I am very happy today")
        self.assertIn("Joy", result)
        self.assertIn("Dominant Emotion", result)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_anger(self, mock_post):
        mock_post.return_value = self._mock_response(
            emotions={"anger": 0.90, "joy": 0.05, "sadness": 0.05}
        )
        result = emotion_detector("I am really mad about this")
        self.assertIn("Anger", result)
        self.assertIn("Dominant Emotion", result)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_disgust(self, mock_post):
        mock_post.return_value = self._mock_response(
            emotions={"disgust": 0.80, "joy": 0.10, "sadness": 0.10}
        )
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertIn("Disgust", result)
        self.assertIn("Dominant Emotion", result)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_sadness(self, mock_post):
        mock_post.return_value = self._mock_response(
            emotions={"sadness": 0.85, "joy": 0.10, "anger": 0.05}
        )
        result = emotion_detector("I am so sad about this")
        self.assertIn("Sadness", result)
        self.assertIn("Dominant Emotion", result)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_fear(self, mock_post):
        mock_post.return_value = self._mock_response(
            emotions={"fear": 0.80, "joy": 0.10, "anger": 0.10}
        )
        result = emotion_detector("I am really afraid that this will happen")
        self.assertIn("Fear", result)
        self.assertIn("Dominant Emotion", result)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_server_error(self, mock_post):
        mock_post.return_value = self._mock_response(status_code=500, emotions={})
        result = emotion_detector("This should fail")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
