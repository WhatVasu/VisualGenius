import os
import google.generativeai as gemini

from PIL import Image
import requests
from io import BytesIO

def get_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img

def get_details(url):
    api = "AIzaSyBVOpDlIL0s-qWRfRl1QM9LBc9FhmAlmDo"
    api_key = os.getenv("GOOGLE_API_KEY")
    gemini.configure(api_key=api)
    model=gemini.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content(get_image_from_url(url))
    return response.text


