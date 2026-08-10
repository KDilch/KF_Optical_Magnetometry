import unittest
import numpy as np
from freq_inference import freq_from_fft, freq_from_periodogram, freq_from_autocorr, ipFFT


class TestFrequencyInferenceUtilities(unittest.TestCase):
    def setUp(self):
        self.fs = 1000.0  # Sampling rate 1kHz
        self.duration = 2.0  # 2 seconds
        self.t = np.arange(0, self.duration, 1.0 / self.fs)
        self.target_freq = 29.0  # 29 Hz target frequency

        # Generate simple sine wave: y = sin(2 * pi * f * t)
        self.signal = np.sin(2.0 * np.pi * self.target_freq * self.t)

    def test_freq_from_fft(self):
        # We test both with and without interpolation
        est_freq = freq_from_fft(self.signal, self.fs, window_name=None)
        # Note: freq_from_fft returns frequency in rad/s, let's verify:
        # In freq_from_fft: x_fft = 2 * pi * fftfreq(len, 1/fs)
        # So yes, it returns angular frequency! 2 * pi * f
        self.assertAlmostEqual(est_freq, 2.0 * np.pi * self.target_freq, delta=0.5 * 2.0 * np.pi)

        est_freq_interp = freq_from_fft(self.signal, self.fs, window_name=None, interpolation=True)
        self.assertAlmostEqual(est_freq_interp, 2.0 * np.pi * self.target_freq, delta=0.1 * 2.0 * np.pi)

    def test_freq_from_periodogram(self):
        # Welch needs a longer signal for resolution, use 5 seconds
        t_long = np.arange(0, 5.0, 1.0 / self.fs)
        signal_long = np.sin(2.0 * np.pi * self.target_freq * t_long)
        # Returns angular frequency 2 * pi * f
        est_freq = freq_from_periodogram(signal_long, self.fs)
        self.assertAlmostEqual(est_freq, 2.0 * np.pi * self.target_freq, delta=3.0 * 2.0 * np.pi)

    def test_freq_from_autocorr(self):
        # Returns angular frequency 2 * pi * f
        est_freq = freq_from_autocorr(self.signal, self.fs)
        self.assertAlmostEqual(est_freq, 2.0 * np.pi * self.target_freq, delta=1.5 * 2.0 * np.pi)

    def test_ipfft(self):
        # Returns frequency in Hz!
        est_freq = ipFFT(self.signal, self.fs, fmax=50.0)
        self.assertAlmostEqual(est_freq, self.target_freq, delta=1.0)


if __name__ == '__main__':
    unittest.main()
