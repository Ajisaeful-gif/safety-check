from flask import Flask, render_template, request, Response, jsonify, session
import tensorflow as tf
from werkzeug.utils import secure_filename
from image_detection import process_image
from video_detection import video_detection
from detection_webcam import detect_objects
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'broyden1'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = 'static/results'

class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = detect_objects(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/deteksi',methods=['GET', 'POST'])
def deteksi():
    session.clear()
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                img = cv2.imread(image_path)
                result_img = process_image(img)
                result_filename = 'result.jpg'
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                cv2.imwrite(result_path, result_img)
                return render_template('deteksi.html', image_path=image_path, result_path=result_filename)
    return render_template('deteksi.html')
    
@app.route('/deteksi_video',methods=['GET', 'POST'])
def deteksi_video():
    # session.clear()
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('deteksi_video.html', form=form)


@app.route('/video')
def video():
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/deteksi_webcam', methods=['GET','POST'])
def deteksi_webcam():
    session.clear()
    return render_template('deteksi_webcam.html')

@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
