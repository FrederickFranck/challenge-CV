import os
import pathlib
#import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = pathlib.Path(__file__).parent / "static/uploads"
CLASSES = ['nv', 'bkl', 'mel', 'akiec', 'bcc', 'df', 'vasc']


model = load_model(pathlib.Path(__file__).parent / "model/model_h.h5")

# Create app & routes
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Default API route
@app.route("/", methods=["GET","POST"])
def route_api():
    if request.method == "GET":
        
        return render_template("index.html")

    if request.method == "POST":
        if 'file' not in request.files:
            flash("No image uploaded")
            return render_template("index.html")
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template("index.html")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'] / filename)
            print(f"path = {path}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            
            #img = Image.open(file)
            _img = image.load_img(path, target_size=(224,224))
            
            img = image.img_to_array(_img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)

            pred = model.predict(img)
            print(f"{type(pred)} : {pred}")
            p = np.round(pred,2)
            print(f"{type(p)} : {p}")
            

        return render_template("index.html",user_image=f"uploads/{filename}",predictions=p[0],classes=CLASSES)


if __name__ == "__main__":
    # You want to put the value of the env variable PORT if it exist (some services only open specifiques ports)
    port = int(os.environ.get("PORT", 5000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost will only be reachable from inside de server.
    app.run(host="0.0.0.0", threaded=True, port=port, debug=True)