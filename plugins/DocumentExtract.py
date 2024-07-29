import os
from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\vasu dev\PycharmProjects\VisionWeb\visionwebproject-cf247d99aceb.json'
from deep_translator import GoogleTranslator
def gtranslator(text,dest='en'):
  root=GoogleTranslator(source="auto",target=dest)
  result=root.translate(text)
  return result
def detect_document_text(image):
    client = vision.ImageAnnotatorClient()
    try:
        response = client.document_text_detection(image=image)
        full_text = ""
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        full_text += word_text + " "
                full_text += "\n"
            full_text += "\n"
        if response.error.message:
            raise Exception(response.error.message)
        return full_text
    except Exception as e:
        return str(e)
