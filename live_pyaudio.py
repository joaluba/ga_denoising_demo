
import sounddevice as sd
import torch
import sys
import time
sys.path.append('/Users/joanna.luberadzka/Projects/denoiser_realtime/')
import causal_improved_sudormrf_v3 
import pyaudio
import numpy as np


# Choose device to compute: 
device="mps" 
# device="cpu"

# Load pytorch model
model = causal_improved_sudormrf_v3.CausalSuDORMRF(
    in_audio_channels=1,
    out_channels=512,
    in_channels=256,
    num_blocks=16,
    upsampling_depth=5,
    enc_kernel_size=21,
    enc_num_basis=512,
    num_sources=1,
    )

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('e39_sudo_whamr_16k_enhnoisy_augment.pt', map_location=device))
model = model.module.to(device)
model.eval()

CHUNK_SIZE = 2048  # Number of frames per buffer
N_BATCHES=1 # number of smaller batches the frame is divided to
FORMAT = pyaudio.paFloat32  # format 


p = pyaudio.PyAudio()

# Open the input audio stream
input_stream = p.open(
    format=FORMAT,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
    input_device_index=0
)

# Open the output audio stream
output_stream = p.open(
    format=FORMAT,
    channels=1,
    rate=16000,
    output=True,
    frames_per_buffer=CHUNK_SIZE,
    output_device_index=1
)


# Start the audio streams
input_stream.start_stream()
output_stream.start_stream()

try:
    while True:
        # print(output_stream.get_write_available())
        # Read audio input
        data = input_stream.read(CHUNK_SIZE)
        
        # Convert the binary data to a numpy array
        audio = np.frombuffer(data, dtype=np.float32)
        
        # Convert the numpy array to a PyTorch tensor
        audio_tensor = torch.from_numpy(audio)
   
        # Make a batch with a few smaller frames
        audio_tensor = torch.stack(torch.split(audio_tensor, len(audio_tensor) // N_BATCHES))
    
        # Reshape the tensor to match the expected input shape of the model 
        audio_tensor = audio_tensor.unsqueeze(1)
        
        # Preprocess audio frame (normalization)
        ini_nrg = torch.sum(audio_tensor ** 2)
        audio_tensor = (audio_tensor - torch.mean(audio_tensor)) / torch.std(audio_tensor)
        
        # Send the tensor to the same device as the model
        audio_tensor=audio_tensor.to(device)

        # Pass the audio tensor through the model
        output = model(audio_tensor)
        
        # Post-processing on the output
        output /= torch.sqrt(torch.sum(output ** 2) / ini_nrg) #energy constraint
        output=output.squeeze(1)
        
        # Put small batches back into one frame
        output=output.reshape(output.shape[0]*output.shape[1])
        output=output.unsqueeze(1)

        # Convert the output tensor back to a numpy array if needed
        output_array = output.detach().cpu().numpy()

        # Processed audio can be written back to the stream for audio output
        output_stream.write(output_array.tobytes())

except KeyboardInterrupt:
    pass

input_stream.stop_stream()
output_stream.stop_stream()
input_stream.close()
output_stream.close()

# Terminate PyAudio
p.terminate()