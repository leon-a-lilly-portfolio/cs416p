import numpy as np
from scipy.io import wavfile

samplerate, data = wavfile.read("message.wav")
samples = data.astype(np.float64) / 32768.0 #to rescale to -1..1
samples_per_bit = 160
t = np.linspace(0, samples_per_bit / samplerate, samples_per_bit, endpoint=False)

cos_2025 = np.cos(2 * np.pi * 2025 * t)
sin_2025 = np.cos(2 * np.pi * 2025 * t)
cos_2225 = np.cos(2 * np.pi * 2225 * t)
sin_2225 = np.cos(2 * np.pi * 2225 * t)

num_bits = len(samples) // samples_per_bit
blocks = samples[:num_bits * samples_per_bit].reshape(num_bits, samples_per_bit)

bits = []
for block in blocks:
    power_2025 = np.dot(block, cos_2025)**2 + np.dot(block, sin_2025)**2
    power_2225 = np.dot(block, cos_2225)**2 + np.dot(block, sin_2225)**2
    bits.append(1 if power_2025 > power_2225 else 0)

bytes_out = []
num_bytes = len(bits) // 10
for i in range(num_bytes):
    byte_bits = bits[i*10 : i*10+10]
    data_bits = byte_bits[1:9]
    value = 0
    for j, bit in enumerate(data_bits):
        value |= bit << j
    bytes_out.append(value)

message = ''.join(chr(b) for b in bytes_out)
with open("message.txt", "w") as file:
    file.write(message)
print(message)
