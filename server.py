from flask import render_template
from flask import Flask, jsonify
from flask_cors import CORS
from keras.engine.saving import load_model
import pickle
import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
from utils import crossdomain, cols_list
from flask import request

app = Flask(__name__)
CORS(app)

# Load the model
# model = pickle.load(open('classifier_model.pkl', 'rb'))
# load model\
global model
model = load_model('classifier_model.h5')
global graph
graph = tf.get_default_graph()

@app.route('/')
@crossdomain(origin='*')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['get'])
@crossdomain(origin='*')
def predict():
    feature1 = request.args.get('feature1', None)
    feature2 = request.args.get('feature2', None)
    feature3 = request.args.get('feature3', None)
    feature4 = request.args.get('feature4', None)
    feature5 = request.args.get('feature5', None)
    feature6 = request.args.get('feature6', None)
    feature7 = request.args.get('feature7', None)
    feature8 = request.args.get('feature8', None)
    feature9 = request.args.get('feature9', None)
    feature10 = request.args.get('feature10', None)
    feature11 = request.args.get('feature11', None)
    feature12 = request.args.get('feature12', None)

    data = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
            feature11, feature12]
    # one hot encode tha incoming data
    new_data = []
    try:
        for i in range(12):
            pkl_file = open('encoders/encoder_' + cols_list[i] + '.pkl', 'rb')
            encoder_file = pickle.load(pkl_file)
            new_data.append(encoder_file.transform([data[i]])[0])
            pkl_file.close()

        # load scaler model
        scaler = joblib.load("scaler.save")
        x = np.array(new_data).reshape(1, -1)
        new_data_scaled = scaler.transform(x)
        # predict
        with graph.as_default():
            result = model.predict(new_data_scaled)

        if result[0][0] < result[0][1]:
            status = 'customer is ACTIVE'
        else:
            status = 'customer is INACTIVE'
        print("\n\n\n")
        print(result)
        print("\n\n\n")
    except:
        status = 'Please enter correct feature'
    data = {'json_key_for_the_prediction': status}

    return jsonify(data)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
