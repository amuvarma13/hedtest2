import numpy as np

def convolve1D(data, window_size=10):

    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    # Initialize the output array
    num_rows = data.shape[0] - window_size + 1
    num_cols = data.shape[1]
    smoothed_data = np.zeros((num_rows, num_cols))

    for i in range(num_cols):
        # Use np.convolve to apply the moving average
        smoothed_data[:, i] = np.convolve(data[:, i], np.ones(window_size) / window_size, mode='valid')

    return smoothed_data