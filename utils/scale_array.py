import numpy as np

def create_cubic_array(L, n):
    if 2 * n > L:
        raise ValueError("n must be no more than L/2 to leave room for the plateau at 1.")
    
    # Generate x values for cubic rise and fall
    x_rise = np.linspace(0, 1, n)
    x_fall = np.linspace(1, 0, n)

    # Cubic function rise to 1 and fall to 0
    cubic_rise = x_rise**3
    cubic_fall = x_fall**3

    # Middle section at 1
    middle_section = np.ones(L - 2 * n)

    # Combine all parts
    full_array = np.concatenate((cubic_rise, middle_section, cubic_fall))

    return full_array

def scale_array(array, scale_index=20):
    length = array.shape[0]
    result_array = create_cubic_array(length, scale_index)
    result_array_reshaped = result_array[:, np.newaxis]
    scaled_array = array * result_array_reshaped
    return scaled_array

