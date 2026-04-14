import numpy as np
import sounddevice as sd
from scipy.io import wavfile

samplerate = 48000
frequency = 440
duration = 1.0
amplitude_1_4 = 8192
amplitude_1_2 = 16384

t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)

sine_arr = (amplitude_1_4 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
wavfile.write("sine.wav", samplerate, sine_arr)

clipped_arr = (amplitude_1_2 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
clipped = np.clip(clipped_arr, -8192, 8192)
wavfile.write("clipped.wav", samplerate, clipped)

sd.play(clipped, samplerate)
sd.wait()
