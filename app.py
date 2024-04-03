import numpy as np
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer
from util import base64_to_pil



app = Flask(__name__)

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:3020/')

'''
MODEL_PATH = './models/your_model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
model._make_predict_function() 
print('Model loaded. Start serving...')
'''


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful input
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        preds = model_predict(img, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))  # Max probability
        pred_class = decode_predictions(preds, top=1)  # ImageNet Decode

        result = str(pred_class[0][0][1])  # Convert to string
        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 3020), app)
    http_server.serve_forever()
