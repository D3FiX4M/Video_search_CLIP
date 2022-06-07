from flask import Flask, render_template
from PIL import Image
import flask
import os
import cv2
from werkzeug.utils import redirect

from lib.Preprocessing import preprocess
from lib import Feature_map, Video_into_frames, Search_frame
from lib.Video_into_frames import split_frame

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/upload-image', methods=['POST', 'GET'])
def uploadImage():
    form = flask.request.form
    discription = form['discription']
    top = form['count_top']
    if top:
        top = int(top)
    else:
        top = 1
    files = flask.request.files.getlist("file[]")
    images = []
    dir_images = []
    for idx, file in enumerate(files):
        image = Image.open(file)
        image.save(f'static/tmp/{idx}.png')
        dir_images.append(f'tmp/{idx}.png')

        image = preprocess(image)
        images.append(image)

    image_features = Feature_map.image_map_features(images)
    text_features = Feature_map.text_map_features([discription])

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    result = Search_frame.find_top_frames(dir_images, text_features, image_features, top=top)

    return render_template('imageView.html', resultPaths = list(result), result=result)

@app.route('/upload-video', methods=['POST', 'GET'])
def uploadVideo():
    form = flask.request.form
    discription = form['discription']
    top = form['count_top']
    fps = form['fps']
    if fps:
        fps = int(fps)
    else:
        fps = 2
    if top:
        top = int(top)
    else:
        top = 1
    file = flask.request.files["file"]
    file.save('static/videoTmp/video.mp4')
    split_frame('static/videoTmp/video.mp4', 'static/videoTmp', fps)
    os.remove('static/videoTmp/video.mp4')
    images = []
    dir_images = os.listdir('static/videoTmp')
    for idx, file in enumerate(dir_images):
        image = Image.open(os.path.join('static/videoTmp', file))
        image = preprocess(image)
        images.append(image)
        dir_images[idx] = f'videoTmp/{file}'

    image_features = Feature_map.image_map_features(images)
    text_features = Feature_map.text_map_features([discription])

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    result = Search_frame.find_top_frames(dir_images, text_features, image_features, top=top)

    return render_template('imageView.html', resultPaths = list(result), result=result)

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
