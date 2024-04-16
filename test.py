from keras.models import load_model
from tcn import TCN 
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import torch
from keras.models import Sequential
from tensorflow.keras.layers import Dense

# Manually recreate the architecture
input_shape = (None, 768)  # None signifies variable sequence lengths, 768 might be your feature dimension
output_units = 5

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


# Load the weights
# model.load_weights('model_weights.h5')

# print("model weights loaded")
# processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
# model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
# model = model.to(device)
# inputs = processor(resampled_audio, sampling_rate=16000, return_tensors="pt", padding=True)
# inputs = inputs.to(device)
# with torch.no_grad():
#     outputs = model(**inputs)
# output_tensor = outputs.last_hidden_state
# print(output_tensor)