import os
from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\vasu dev\PycharmProjects\VisionWeb\visionwebproject-cf247d99aceb.json'
def detect_logos_uri(image):
    client = vision.ImageAnnotatorClient()
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    return logos