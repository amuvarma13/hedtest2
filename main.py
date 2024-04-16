from flask import Flask, jsonify
import numpy as np
from flask_cors import CORS
# from scale_array import scale_array
# Add Flask CORS

from utils.get_bins import get_output_tensor
from utils.run_inference import run_inference
from utils.load_wav_as_numpy import load_wav_as_numpy
from utils.convolved1D import convolve1D
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')


# app = Flask(__name__)
# CORS(app)


# @app.route('/ping', methods=['GET'])
# def ping():
#     return "pong"



# @app.route('/', methods=['GET'])
def get_data():

    wavs = load_wav_as_numpy("hed4.wav")
    print(f'wavs: {wavs.shape}')

    bins = get_output_tensor(wavs)
    print(f'bins: {bins.shape}')
    predictions = run_inference(bins)
    print(f'predictions: {predictions.shape}')
    smoothed_predictions = predictions.squeeze()
    smoothed_predictions = convolve1D(smoothed_predictions, 3) # data, window
    smoothed_predictions = convolve1D(smoothed_predictions, 3)
    smoothed_predictions = convolve1D(smoothed_predictions, 6)
    smoothed_predictions = convolve1D(smoothed_predictions, 6)
    smoothed_predictions = convolve1D(smoothed_predictions, 6)
    smoothed_predictions = convolve1D(smoothed_predictions, 12)
    smoothed_predictions = convolve1D(smoothed_predictions, 12)
    smoothed_predictions = convolve1D(smoothed_predictions, 12)
    print(f'predictions: {smoothed_predictions.shape}')
    predictions_list = predictions.T.tolist()


    return jsonify(predictions_list)

get_data()



    # shape = dataset.shape[0]
    # emphasize = generate_strong_movement(shape, 200, 2, 100, 20)

    # combined = dataset*0.5 + emphasize
    # scaled_array = scale_array(combined)

    # generate_time_series_data("sd.wav")

    # return jsonify(scaled_array.T.tolist())


# if __name__ == '__main__':
#     app.run(port=8080, host='0.0.0.0')
