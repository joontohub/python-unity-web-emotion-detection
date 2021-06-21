import os
import io
import sys
from flask import Flask
from flask import request, redirect, url_for, send_from_directory, jsonify, json
from werkzeug.utils import secure_filename
from PIL import Image
import base64
#import picture_detector


UPLOAD_FOLDER = '../Images'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav'])


result_emotion = ""
result_prob = 0


def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS


# @app.route('/' , methods=['GET', 'POST'])
# def home():
#     if request.method == "POST":
#         data = request.files['myimage'].read()
#         imag = Image.open(io.BytesIO(data))
#         filename = 'myimage'
#         imag.save(os.path.join(app.root_path, filename))

#         print(type(data))
#         filename = "abc"
#         data.save(os.path.join("/simplePyweb/server/",filename))

#     return 'Hello111, World!'
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            print("activate picture emotion detector")
            result_emotion, result_prob = picture_detector.Detector()
            result_prob = str(result_prob)
            tag = "T"
            data = tag + result_emotion + tag + result_prob
            response = app.response_class(
                response=json.dumps(data),
                status=200,
                mimetype='application/json'
            )

            print("json data :::: ", response)
            return response
            # for browser, add 'redirect' function on top of 'url_for'
            # return url_for("result_page", data=data)
    else:
        return "Hello world it is for post "


if __name__ == '__main__':
    app.run(debug=True)
