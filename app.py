import os
import cv2
import requests
from flask import Flask, render_template, request, redirect, url_for, Response
from google.cloud import vision
from plugins import LandmarkDetection, objectDetection, DocumentExtract, LogoDetect, CelebDetection, ImageDescriber

app = Flask(__name__)
os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\vasu dev\PycharmProjects\VisionWeb\visionwebproject-cf247d99aceb.json'

# Set and ensure the UPLOAD_FOLDER exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_image_path(image_file):
    filename = image_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)
    return filepath


def create_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return vision.Image(content=response.content)
    except requests.exceptions.RequestException as e:
        return str(e)


def create_image_from_file(image_file):
    image_path = get_image_path(image_file)
    with open(image_path, 'rb') as img_file:
        content = img_file.read()
    return vision.Image(content=content), url_for('static', filename=f'uploads/{os.path.basename(image_path)}')


def capture_by_frames():
    global cap
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minSize=(30, 30), minNeighbors=5,
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/face_detect')
def faceindex():
    return render_template('face_detect.html')


@app.route('/start', methods=['POST'])
def start():
    return render_template('face_detect.html')


@app.route('/stop', methods=['POST'])
def stop():
    if cap.isOpened():
        cap.release()
    return render_template('face_detect2.html')


@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index')
def indexx():
    return render_template('index.html')


@app.route('/landmark', methods=['POST', 'GET'])
def landmark_route():
    if request.method == "POST":
        image_url = request.form.get('image_url')
        image_file = request.files.get('image_file')

        if image_file and image_file.filename != '':
            image, uploaded_image_url = create_image_from_file(image_file)
            image_url = None
        elif image_url:
            image = create_image_from_url(image_url)
            uploaded_image_url = None
        else:
            return render_template('landmark.html', error="No image provided.")

        if isinstance(image, str):
            return render_template('landmark.html', error=image)

        results = LandmarkDetection.detect_landmarks(image)
        if isinstance(results, str):
            return render_template('landmark.html', error=results)

        desc = LandmarkDetection.landmarkwiki(results)
        return render_template('landmark.html', land_results=results, info=desc, n=len(desc), image_url=image_url,
                               uploaded_image_url=uploaded_image_url)
    else:
        return render_template('landmark.html', land_results=None)


@app.route('/objec', methods=['POST', 'GET'])
def localize_objects_route():
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        image_file = request.files.get('image_file')

        if image_file and image_file.filename != '':
            image, uploaded_image_url = create_image_from_file(image_file)
            image_url = None
        elif image_url:
            image = create_image_from_url(image_url)
            uploaded_image_url = None
        else:
            return render_template('object_local.html', error="No image provided.")

        if isinstance(image, str):  # check if there's an error message
            return render_template('object_local.html', error=image)

        results = objectDetection.localize_objects_uri(image)
        image_with_boxes = objectDetection.draw_bounding_boxes(image,image_url, results)
        return render_template('object_local.html', object_results=results, image_url=image_url,
                               image_data=image_with_boxes, uploaded_image_url=uploaded_image_url)
    else:
        return render_template('object_local.html', object_results=None)


@app.route('/logo', methods=['POST', 'GET'])
def logo_route():
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        image_file = request.files.get('image_file')

        if image_file and image_file.filename != '':
            image, uploaded_image_url = create_image_from_file(image_file)
            image_url = None
        elif image_url:
            image = create_image_from_url(image_url)
            uploaded_image_url = None
        else:
            return render_template('logo.html', error="No image provided.")

        if isinstance(image, str):
            return render_template('logo.html', error=image)

        results = LogoDetect.detect_logos_uri(image)
        return render_template('logo.html', logo_results=results, image_url=image_url,
                               uploaded_image_url=uploaded_image_url)
    else:
        return render_template('logo.html', logo_results=None)


@app.route('/doctext', methods=['POST', 'GET'])
def doc_text_route():
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        image_file = request.files.get('image_file')
        if image_file and image_file.filename != '':
            image, uploaded_image_url = create_image_from_file(image_file)
            image_url = None
        elif image_url:
            image = create_image_from_url(image_url)
            uploaded_image_url = None
        else:
            return render_template('doctext.html', error="No image provided.")

        if isinstance(image, str):
            return render_template('doctext.html', error=image)

        results = DocumentExtract.detect_document_text(image)
        translated = DocumentExtract.gtranslator(results)
        return render_template('doctext.html', trans=translated, text_results=results, image_url=image_url,
                               uploaded_image_url=uploaded_image_url)
    else:
        return render_template('doctext.html', text_results=None)


@app.route('/Celebdetect', methods=['POST', 'GET'])
def celebdetect():
    web = ''
    img_str = ''
    if request.method == 'POST':
        img_url = request.form['image_url']
        results = CelebDetection.detect_faces_uri(img_url)
        web = results[0]
        img_str = results[1]
        return render_template('/Celebdetect.html', web_entities=web, img_str=img_str)
    return render_template('/Celebdetect.html', web_entities=None, img_str=None)


@app.route('/image description', methods=['POST', 'GET'])
def img_info_route():
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        image_file = request.files.get('image_file')

        if image_file and image_file.filename != '':
            image, uploaded_image_url = create_image_from_file(image_file)
            image_url = None
        elif image_url:
            image = create_image_from_url(image_url)
            uploaded_image_url = None
        else:
            return render_template('image description.html', error="No image provided.")

        if isinstance(image, str):
            return render_template('image description.html', error=image)

        results = ImageDescriber.get_details(image_url)
        return render_template('image description.html', info_results=results, image_url=image_url,
                               uploaded_image_url=uploaded_image_url)
    else:
        return render_template('image description.html', info_results=None)


if __name__ == '__main__':
    app.run(debug=True)
