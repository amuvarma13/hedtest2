from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
aud_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
aud_model = aud_model.to(device)

def get_output_tensor(audio):

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = aud_model(**inputs)
    output_tensor = outputs.last_hidden_state.cpu().numpy()
    return output_tensor