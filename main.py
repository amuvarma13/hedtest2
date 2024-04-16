from flask import Flask, jsonify
import numpy as np
from flask_cors import CORS

from generate_strong_movement import generate_strong_movement
from scale_array import scale_array
from generate_time_series_data import predict
# Add Flask CORS


app = Flask(__name__)
CORS(app)



@app.route('/', methods=['GET'])
def get_data():

    wavs = np.load("head.wav")
    wavs = predict(wavs)
    print(wavs)

    return "True"


    # shape = dataset.shape[0]
    # emphasize = generate_strong_movement(shape, 200, 2, 100, 20)

    # combined = dataset*0.5 + emphasize
    # scaled_array = scale_array(combined)

    # generate_time_series_data("sd.wav")

    # return jsonify(scaled_array.T.tolist())


if __name__ == '__main__':
    app.run(port=8080)
