# pip install azure-cognitiveservices-vision-customvision

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# Now there is a trained endpoint that can be used to make a prediction

ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com"


predictor = CustomVisionPredictionClient("Prediction key", endpoint=ENDPOINT)

# Open the sample image and get back the prediction results.
with open("test.jpg", mode="rb") as test_data:
    results = predictor.predict_image("Project.id", test_data)

# Display the results.
for prediction in results.predictions:
    print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100), prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)
