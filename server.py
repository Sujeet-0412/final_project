"""
Flask server for emotion detection.
"""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

# Initialize Flask app
app = Flask(__name__)


@app.route("/emotionDetector")
def emotion_detector_route():
    """
    Analyze emotions in text passed via query parameter.
    """
    text_to_analyze = request.args.get("textToAnalyze")

    if not text_to_analyze:
        return "No text provided! Please supply textToAnalyze as a query parameter."

    # Call the emotion detector function
    response = emotion_detector(text_to_analyze)

    # Handle invalid input or server error
    if response is None:
        return "Invalid input or server error! Try again."

    # Return the formatted response string
    return response


@app.route("/")
def render_index_page():
    """
    Render the index page.
    """
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
