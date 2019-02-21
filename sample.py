# pip install azure-cognitiveservices-vision-customvision

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# Now there is a trained endpoint that can be used to make a prediction
#prediction_key = "d03be561c2cc43a7a6845cbc58b2dbdf"
ENDPOINT = "https://southcentralus.api.cognitive.microsoft.com"
#project.id = "822f64ef-8867-4812-810e-6869b71ec4f4"
#iteration.id = "94a54977-701c-461c-8461-8273afe75f6e"

predictor = CustomVisionPredictionClient("d03be561c2cc43a7a6845cbc58b2dbdf", endpoint=ENDPOINT)

# Open the sample image and get back the prediction results.
with open("dji1542012408189.jpg", mode="rb") as test_data:
    results = predictor.predict_image("822f64ef-8867-4812-810e-6869b71ec4f4", test_data, "94a54977-701c-461c-8461-8273afe75f6e")

# Display the results.
for prediction in results.predictions:
    print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100), prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)
