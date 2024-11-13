import sounddevice as sd
import numpy as np

# Record
duration = 5  # Record time (sec)
sampling_rate = 44100  # Hz

print("Record...")
audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='int16')
sd.wait()  # Wait until recording ends
print("Recording complete.")

# Convert float data (range -1 to 1) to int16
float_audio_data = np.random.rand(44100) * 2 - 1
int_audio_data = (float_audio_data * 32767).astype(np.int16)

sd.play(audio_data, samplerate=44100)  # Play by setting the sampling rate
sd.wait()
