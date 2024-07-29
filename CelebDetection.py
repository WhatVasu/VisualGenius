import os
import requests
from google.cloud import vision
from PIL import Image, ImageDraw
from io import BytesIO
import base64
def detect_faces_uri(uri):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\vasu dev\PycharmProjects\VisionWeb\visionwebproject-cf247d99aceb.json'
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.face_detection(image=image)
    web_response = client.web_detection(image=image)
    faces = response.face_annotations
    web_entitie = web_response.web_detection.web_entities
    web = []
    for i in web_entitie:
        if round(i.score) > 1:
            web.append(i)
    response = requests.get(uri)
    img = Image.open(BytesIO(response.content))
    draw = ImageDraw.Draw(img)
    for face in faces:
        vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(vertices + [vertices[0]], width=5, fill='red')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return  web, img_str
