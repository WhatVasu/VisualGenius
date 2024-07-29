import os
from google.cloud import vision
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\vasu dev\PycharmProjects\VisionWeb\visionwebproject-cf247d99aceb.json'
def localize_objects_uri(image):
    client=vision.ImageAnnotatorClient()
    objects = client.object_localization(image=image).localized_object_annotations

    results = []
    for object_ in objects:
        bounding_box = [(vertex.x, vertex.y) for vertex in object_.bounding_poly.normalized_vertices]
        object_info = {
            "name": object_.name,
            "confidence": str(object_.score.__round__(2))[2:4] + "%",
            "bounding_box": bounding_box
        }
        results.append(object_info)

    return results


def draw_bounding_boxes(Immage,uri, results):
    if uri!=None:
        response = requests.get(uri)
        image = Image.open(BytesIO(response.content))
        draw = ImageDraw.Draw(image)
        width, height = image.size
    else:
        draw = ImageDraw.Draw(Immage)
        width, height = Immage.size
    for obj in results:
        bounding_box = obj['bounding_box']
        box = [(int(vertex[0] * width), int(vertex[1] * height)) for vertex in bounding_box]
        draw.polygon(box, outline='red', width=2)
        draw.text(box[0], obj['name'], fill='green')

    if uri != None:
      buffered = BytesIO()
      image.save(buffered, format="PNG")
      img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        buffered = BytesIO()
        Immage.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str