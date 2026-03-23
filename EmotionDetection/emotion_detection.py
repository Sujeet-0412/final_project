import requests
import json

def emotion_detector(text_to_analyze):
    # URL of the sentiment analysis service
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    
    # Create a dictionary with the text to be analyzed
    myobj = { "raw_document": { "text": text_to_analyze } }
    
    # Set the headers required for the API request
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    
    # Send a POST request to the API with the text and headers
    response = requests.post(url, json=myobj, headers=headers)

    # Parse the JSON response
    formatted_response = response.json()

    if response.status_code == 200: 
        label = formatted_response['emotionPredictions'][0]['emotion'] 
        emotions_output = "\n".join(
            [f"{emotion.capitalize()}: {score:.3f}" for emotion, score in label.items()]
        )
        dominant_emotion = max(label, key=label.get)
        dominant_score = label[dominant_emotion]

        result = (
            f"{emotions_output}\n\n"
            f"Dominant Emotion:\n{dominant_emotion.capitalize()} with a score of {dominant_score:.3f}"
        )
    elif response.status_code == 500:
        result = None 
    return result

