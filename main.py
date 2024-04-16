from flask import Flask, jsonify
import numpy as np
from flask_cors import CORS
# from scale_array import scale_array
# Add Flask CORS

from utils.get_bins import get_output_tensor
from utils.run_inference import run_inference


app = Flask(__name__)
CORS(app)



@app.route('/', methods=['GET'])
def get_data():

    wavs = np.load("hed.wav")

    bins = get_output_tensor(wavs)
    print(f'bins: {bins.shape}')
    prediction = run_inference(bins)

    return "True"


    # shape = dataset.shape[0]
    # emphasize = generate_strong_movement(shape, 200, 2, 100, 20)

    # combined = dataset*0.5 + emphasize
    # scaled_array = scale_array(combined)

    # generate_time_series_data("sd.wav")

    # return jsonify(scaled_array.T.tolist())


if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')
