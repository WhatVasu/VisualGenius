import requests
from bs4 import BeautifulSoup
import os
from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\vasu dev\PycharmProjects\VisionWeb\visionwebproject-cf247d99aceb.json'
def landmarkwiki(li):
    innfo = []
    for i in li:
        try:
            response = requests.get(f"https://en.wikipedia.org/wiki/{i.description}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.find('div', class_='mw-content-ltr mw-parser-output')
            if content_div:
                paragraphs = content_div.find_all('p')
                if paragraphs:
                    info = ' '.join(paragraph.text for paragraph in paragraphs[:2])
                    innfo.append(info)
                else:
                    innfo.append(f"No content found for {i.description}")
            else:
                innfo.append(f"No content div found for {i.description}")
        except requests.exceptions.RequestException as e:
            innfo.append(f"Error fetching data for {i.description}: {e}")
    return innfo
def create_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return vision.Image(content=response.content)
    except requests.exceptions.RequestException as e:
        return str(e)
def detect_landmarks(image):
    client = vision.ImageAnnotatorClient()

    try:
        response = client.landmark_detection(image=image)
        return response.landmark_annotations
    except Exception as e:
        return str(e)