from scipy import fft
import numpy as np
from scipy.signal import get_window, periodogram


def freq_from_autocorr(signal_arr, fs, window_name=None):
    """Estimate frequency using autocorrelation of the signal

    Args:
        signal_arr (ndarray): Input signal
        fs (float): Sampling rate of the signal
        window (None or string): string with the window name available names are

    Returns:
        float: Estimated frequency in Hz
    """
    windowed_signal = signal_arr
    if window_name:
        window = get_window(window_name, len(signal_arr))
        windowed_signal = signal_arr * window

    # Compute autocorrelation of the signal
    autocorr = np.correlate(windowed_signal, windowed_signal, 'full')[-len(windowed_signal):]

    # find the peak of autocorrelation
    inflection = np.diff(np.sign(np.diff(autocorr)))  # Find the second-order differences
    peaks = (inflection < 0).nonzero()[0] + 1  # Find where they are negative
    delay = peaks[autocorr[peaks].argmax()]  # Of those, find the index with the maximum value

    return 2*np.pi*(fs / delay)


def freq_from_fft(signal_arr, fs, window_name=None, interpolation=False):
    """Estimate frequency using fast Fourier transforms

    Args:
        signal_arr (ndarray): Input signal
        fs (float): Sampling rate of the signal
        window (bool): Use hamming window, check for window types here https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        interpolation (None or string): Interpolate the signal

    Returns:
        float: Estimated frequency in Hz
    """
    windowed_signal = signal_arr
    if window_name:
        window = get_window(window_name, len(signal_arr))
        windowed_signal = signal_arr * window
    if interpolation:
        interp_t = np.linspace(0, 1/fs*len(windowed_signal), 10*len(windowed_signal))
        interp_sign = np.interp(interp_t,
                                np.linspace(0, 1/fs*len(windowed_signal), len(windowed_signal)),
                                windowed_signal)
        y_fft = fft.fft(interp_sign)
        x_fft = 2 * np.pi * fft.fftfreq(10*len(windowed_signal), interp_t[1]-interp_t[0])
        peak = np.argmax(np.abs(y_fft))
    else:
        y_fft = fft.fft(windowed_signal)
        x_fft = 2*np.pi*fft.fftfreq(len(windowed_signal), 1/fs)
        peak = np.argmax(np.abs(y_fft))
    return x_fft[peak]


def freq_from_periodogram(signal_arr, fs, window_name=None):
    """Estimate frequency using periodogram method.

    :param signal_arr: (ndarray) Input signal
    :param fs: (float) Sampling rate
    :param window: (None or string)
    :return:
    """
    windowed_signal = signal_arr
    if window_name:
        window = get_window(window_name, len(signal_arr))
        windowed_signal = signal_arr * window
    freqs, Pxx_den = periodogram(windowed_signal, fs=fs)
    peak = np.argmax(Pxx_den)
    return 2*np.pi*freqs[peak]

