import numpy as np
from scipy.interpolate import interp1d

def extend(array, m):

    n = array.shape[0]
    if n == m:
        return array  # If the current size is the same as the desired size, return the original array
    # Create an array of original indices based on the current array size
    x_old = np.linspace(0, n - 1, n)
    # Create an array of new indices based on the desired size
    x_new = np.linspace(0, n - 1, m)
    # Interpolate each column of the array
    interpolated_array = np.array([interp1d(x_old, array[:, i], kind='linear', fill_value='extrapolate')(x_new) for i in range(5)]).T

    return interpolated_array
