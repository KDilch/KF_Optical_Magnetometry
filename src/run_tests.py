# import numpy as np
# from scipy.fft import fft, fftfreq
# import matplotlib.pyplot as plt
#
#
# dt = 0.01
# time = np.arange(1, 10, dt)
# data = np.array([np.sin(2 * np.pi * 1 * t) for t in time])
# fftdata = fft(data)
# sample_rate = 1 / dt
# num = int(np.size(fftdata))
# freq = fftfreq(num, dt)
# print("frequency", abs(freq[np.where(fftdata == np.amax(fftdata))][0]))
#
# plt.plot(freq, abs(fftdata))
# plt.show()
# SAMPLE_RATE = 44100  # Hertz
# DURATION = 5  # Seconds
#
# def generate_sine_wave(freq, sample_rate, duration):
#     x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
#     frequencies = x * freq
#     # 2pi because np.sin takes radians
#     y = np.sin((2 * np.pi) * frequencies)
#     return x, y
#
# # Generate a 2 hertz sine wave that lasts for 5 seconds
# x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
#
# N = SAMPLE_RATE * DURATION
#
# freq = fftfreq(N, 1/SAMPLE_RATE)
def run__test(*args):
    pass