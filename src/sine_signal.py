import numpy as np
import matplotlib.pyplot as plt
import signal

from freq_inference import ipFFT

# Define the signal parameters
is_ipFFT = True
is_periodogram = False
A = 1.0
f0 = 1.1
phi = 0.0
fs = 100.0
T = 100.0
sigma = 0.1

# Generate the noisy signal
t = np.arange(0, T, 1/fs)
x = A*np.sin(2*np.pi*f0*t + phi) + np.random.normal(scale=sigma, size=len(t))


# Compute the CRB for frequency
def CRB_f(t):
    return (sigma**2*(1/fs)**2)/(A**2*t**3*np.sin(2*np.pi*f0*t)**2)


# Compute the error of frequency estimation over time
f_est_array = np.zeros_like(t)
error_array = np.zeros_like(t)
CRB_f_array = np.zeros_like(t)
for i, ti in enumerate(t):
    if i > 100:
        if is_ipFFT:
            f_est_array[i] = ipFFT(x[0:i], fs=fs, fmax=2.0, win_type='hanning')
        elif is_periodogram:
            f, Pxx = signal.periodogram(x[0:i], fs=fs, window="cosine")
            f_est_array[i] = f[np.argmax(Pxx)]
        else:
            raise ValueError("Unsupported frequency inference method.")

        # save the errors and CRB
        error_array[i] = np.abs(f_est_array[i] - f0)**2
        CRB_f_array[i] = CRB_f(ti)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
ax1.plot(t, f_est_array, label='Estimated frequency')
ax1.axhline(y=f0, color='r', linestyle='--', label='True frequency')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Frequency Estimation over Time')
ax1.legend()
ax2.semilogy(t, error_array, label='Estimation Error')
ax2.semilogy(t, CRB_f_array, '--k', label='CRB')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (Hz)')
ax2.set_title('Frequency Estimation Error over Time')
ax2.legend()
plt.show()
