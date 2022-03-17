from scipy.fft import fft, fftfreq
import numpy as np


def perform_discrete_fft(simulation_params, xs):
    SAMPLE_RATE = 1/simulation_params.dt
    NUM_SAMPLES = int(simulation_params.t_max*SAMPLE_RATE)
    # x_fft = np.abs(fft(x_ekf_est[:, 1]))
    x_fft = np.abs(fft(xs[:, 1]))

    # get the list of frequencies
    frequencies = fftfreq(NUM_SAMPLES, simulation_params.dt)
    return frequencies, x_fft
