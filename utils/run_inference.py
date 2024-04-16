from keras.models import load_model
from tcn import TCN 
from keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch

# Manually recreate the architecture
input_shape = (None, 768)  # None signifies variable sequence lengths, 768 might be your feature dimension
output_units = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Sequential([
    TCN(input_shape=input_shape,
        nb_filters=64,
        kernel_size=24,
        nb_stacks=2,
        dilations=[1, 2, 4],
        padding='causal',
        use_skip_connections=True,
        dropout_rate=0,
        return_sequences=True),  # Ensure this is True for sequence output
    Dense(output_units, activation='linear')  # Output layer with linear activation
])

print(model.summary())
model.load_weights('model_weights.h5')


def run_inference (nd_array):
    
    predictions = model.predict(nd_array)
    return predictions.squeeze()
