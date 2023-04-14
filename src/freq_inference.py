from scipy import fft
import numpy as np
from scipy.signal import get_window, periodogram, welch


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
    freqs, Pxx_den = welch(windowed_signal, fs=fs)
    peak = np.argmax(Pxx_den)
    return 2*np.pi*freqs[peak]


def ipFFT(x, fs, fmax, win_type='hamming'):
    """
    The Interpolated Fast Fourier Transform: A Comparative Study" by J. Schoukens, R. Pintelon, and H. Van hamme suggests that the IP-FFT algorithm should be modified for different window functions.
    Specifically, the paper suggests that the interpolation weights used in the algorithm should be adjusted to account for the spectral leakage caused by different window functions.
    Implements the IP-FFT algorithm for frequency inference with windowing.

    Parameters:
    x (numpy array): Input time-domain signal
    fs (float): Sampling frequency of the input signal in Hz
    fmax (float): Maximum frequency to be estimated in Hz
    win_type (str): Type of window to be applied to the input signal.
                    Supported window types include 'hamming', 'hanning', and 'blackman'.

    Returns:
    freq (float): Estimated frequency of the input signal in Hz
    """

    # Get the length of the input signal
    N = len(x)

    # Apply windowing to the input signal
    if win_type == 'hamming':
        win = np.hamming(N)
        interp_wts = np.array([1.08, 0.926, 0.701, 0.44, 0.198])
    elif win_type == 'hanning':
        win = np.hanning(N)
        interp_wts = np.array([1.123, 1.023, 0.757, 0.456, 0.19])
    elif win_type == 'blackman':
        win = np.blackman(N)
        interp_wts = np.array([1.165, 1.036, 0.72, 0.319, 0.062])
    else:
        raise ValueError(
            "Unsupported window type. Supported window types include 'hamming', 'hanning', and 'blackman'.")

    x = x * win

    # Compute the FFT of the input signal
    X = np.fft.fft(x)

    # Compute the frequency resolution of the FFT
    df = fs / N

    # Compute the number of frequency bins to interpolate
    n_interp = int(fmax / df)

    # Compute the indices of the frequency bins to interpolate
    interp_idx = np.arange(n_interp) + 1

    # Interpolate the FFT to estimate the frequency spectrum
    X_interp = np.zeros(n_interp, dtype=np.complex64)
    for i in range(n_interp):
        w = interp_wts[i % 5]  # Use the appropriate interpolation weight for this bin
        idx = interp_idx[i] * np.arange(5) + 1
        idx = idx[(idx >= 0) & (idx < N)]  # Ensure that the indices are within bounds
        X_interp[i] = np.sum(w * np.sin(np.pi * interp_idx[i] / N) * X[idx])

    # Find the frequency with the maximum magnitude in the interpolated spectrum
    max_idx = np.argmax(np.abs(X_interp))
    freq = max_idx * df

    return freq


