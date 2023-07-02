
import torch
#import sys
#import time
#sys.path.append('/Users/joanna.luberadzka/Projects/denoiser_realtime/')
import causal_improved_sudormrf_v3 
import pyaudio
import numpy as np


# Choose device to compute: 
device="mps" 
# device="cpu"

# Load pytorch model
model = causal_improved_sudormrf_v3.CausalSuDORMRF(
        in_audio_channels=2,
        out_channels=512,
        in_channels=256,
        num_blocks=16,
        upsampling_depth=5,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=1)

#model = torch.nn.DataParallel(model)
#model.load_state_dict(torch.load('e39_sudo_whamr_16k_enhnoisy_augment.pt', map_location=device))
model.load_state_dict(torch.load('m5_alldata_mild_causal.pt', map_location=device))
model.to(device)

# model = model.module.to(device)
model.eval()

CHUNK_SIZE = 4096  # Number of frames per buffer
N_BATCHES=4 # number of smaller batches the frame is divided to
FORMAT = pyaudio.paFloat32  # format (i wold use 16bit if its possible, should be enough)


p = pyaudio.PyAudio()

# Open the input audio stream
input_stream = p.open(
    format=FORMAT,
    channels=2,
    rate=16000,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
    input_device_index=0
)

# Open the output audio stream
output_stream = p.open(
    format=FORMAT,
    channels=2,
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
        audio = np.frombuffer(data, dtype=np.float32) #16 bit should be enough!
        audio = np.vstack((audio[::2], audio[1::2])).T #de-interleave input array
        #https://ijc8.me/2020/04/20/quick-audio-video-python/

        # Convert the numpy array to a PyTorch tensor
        audio_tensor = torch.from_numpy(audio)
        #audio_tensor = torch.reshape(audio_tensor, (CHUNK_SIZE, 2))
        # Make a batch with a few smaller frames
        audio_tensor = torch.stack(torch.split(audio_tensor, len(audio_tensor) // N_BATCHES))
    
        # Reshape the tensor to match the expected input shape of the model 
        #audio_tensor = audio_tensor.unsqueeze(1)
        # Preprocess audio frame (normalization)
        ini_nrg = torch.sum(audio_tensor ** 2)
        audio_tensor = (audio_tensor - torch.mean(audio_tensor)) / torch.std(audio_tensor)

        # Send the tensor to the same device as the model
        audio_tensor = audio_tensor.permute(0, 2, 1)
        audio_tensor=audio_tensor.to(device)

        # Pass the audio tensor through the model
        output = model(audio_tensor)
        #output = audio_tensor #BYPASS MODEL
        
        # Post-processing on the output
        output /= torch.sqrt(torch.sum(output ** 2) / ini_nrg) #energy constraint
        #output=output.squeeze(1)
        # Put small batches back into one frame
        output=output.reshape(output.shape[0]*output.shape[1], -1)
        #output=output.view(-1)
        #output = output.reshape(CHUNK_SIZE*2)

        # Convert the output tensor back to a numpy array if needed
        output_array = output.detach().cpu().numpy()
        interleaved = np.empty((output_array.shape[1]*2), dtype=output_array.dtype)
        interleaved[::2] = output_array[0]  # even-indexed samples come from the left channel
        interleaved[1::2] = output_array[1]  # odd-indexed samples come from the right channel
        #output_array = output_array.flatten()
        # Processed audio can be written back to the stream for audio output
        output_stream.write(interleaved.tobytes())

except KeyboardInterrupt:
    pass

input_stream.stop_stream()
output_stream.stop_stream()
input_stream.close()
output_stream.close()

# Terminate PyAudio
p.terminate()