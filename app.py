import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
from gevent.pywsgi import WSGIServer
from keras.models import load_model
from keras.preprocessing import image
from flask import send_from_directory
import tensorflow as tf


UPLOAD_FOLDER = 'static\\uploads'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('models\model.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        filepath = secure_filename(f.filename)
        image_url=os.path.join(app.config['UPLOAD_FOLDER'], filepath)
        f.save(image_url)

        upload_img = os.path.join(UPLOAD_FOLDER, filepath) 
        img = image.load_img(upload_img, target_size=(100, 100))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = img_array / 255.0

        pred = model.predict(img_preprocessed)

        result = np.argmax(pred)
        label=['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana',
            'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn',
            'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White',
            'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green',
            'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry',
            'Strawberry', 'Tomato', 'Watermelon']
        print(label[result])
        return render_template('predict.html', ans=label[result], image_url=url_for('static', filename='uploads/' + filepath))


if __name__ == '__main__':
    app.run(debug=True)
