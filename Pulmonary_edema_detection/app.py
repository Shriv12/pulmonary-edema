from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model



app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/pulmonary", methods=['GET', 'POST'])
def pulmonaryPage():
    return render_template('home.html')


@app.route("/pulmonarypredict", methods = ['POST', 'GET'])
def pulmonarypredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pulmonary.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('home.html', message = message)
    return render_template('pulmonary_predict.html', pred = pred)
    
if __name__ == '__main__':
	app.run(debug = True)