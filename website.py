import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)
model = load_model('BrainTumor10epochs.h5')
@app.route('/')
def index():
    return render_template('index.html')
@app.route("/upload", methods=["post"])
def upload():
    file = request.files["photo-upload"]

    file_add = 'static/'+ file.filename
    file.save(file_add)
    img = cv2.imread(file_add)
    img=Image.fromarray(img)
    img = img.resize((64, 64))
    img=np.expand_dims(img,axis=0)
    image_data = np.array(img, dtype='uint8') / 255
    output = model.predict(image_data)
    if(output>0.5):
        output=1
    else:
        output=0
    print(file_add)
    return render_template('upload.html',data=output,filepath=file.filename)


@app.route('/result', methods=['post'])
def result():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
