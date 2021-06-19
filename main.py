from flask import Flask, render_template, Response

from flask_bootstrap import Bootstrap

from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
#from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL

import os
app = Flask(__name__)
Bootstrap(app)
#app = Flask(__name__, template_folder='../templates')

# ::: FLASK ROUTES
@app.route('/home')
@app.route('/')
def index():
	# Main Page
	return render_template('home.html')

# MODEL_PATH = 'F:\Major Project Main Folder\MAJOR-WEBSITE-SEHAT SAARTHI\Project\model\DenseNet169-best-model.h5'
# MODEL_PATH = '.model\DenseNet169-best-model.h5'


# Load your trained model
model = load_model("DenseNet169-best-model.h5")
model.make_predict_function() 

print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
    '''
    	Args:
    		-- img_path : an URL path where a given image is stored.
    		-- model : a given Keras CNN model.
    '''

    IMG = image.load_img(img_path,color_mode='rgb')
    print(type(IMG))

	# Pre-processing the image
    IMG_ = IMG.resize((224, 224))
    print(type(IMG_))
    IMG_ = np.asarray(IMG_)
    print(IMG_.shape)

    IMG_=np.array([IMG_])

    IMG_ = np.true_divide(IMG_, 255)
    IMG_ = IMG_.reshape(-1, 224, 224, 3)
    print(type(IMG_), IMG_.shape)

    print(model)

    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    prediction = model.predict(IMG_)

    return prediction




@app.route('/predict', methods=['GET', 'POST'])
def upload():

    # Constants:
    classes = {'TRAIN': ['normal', 'Pneumonia', 'COVID']}

    if request.method == 'POST':

		# Get the file from post request
        f = request.files['file']

		# Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

		# Make a prediction
        predictions = model_predict(file_path, model)

        prediction = predictions.tolist()

        print(prediction)
        print(type(prediction))
        print(prediction[0][0])

        predicted_class = predictions.argmax(axis=-1)
        #print("adadadada",predicted_class)
        
        # predicted_class = classes['TRAIN'][prediction[0]
        predicted_class1 = classes['TRAIN'][predicted_class[0]]
        print('We think that is {}.'.format(predicted_class1.lower()))

        return str(predicted_class1).lower()

if __name__ == '__main__':
    # app.run(host='127.0.0.1', debug=True)
    app.run(debug=True)
