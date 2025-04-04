import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from freq_inference import ipFFT

# Define the signal parameters
is_ipFFT = False
is_periodogram = True
A = 10.0
omega_0 = 50.2
phi = 0.0
fs = 100.0
dt = 1./fs
T = 400.0
T2 = 100.0
sigma = 1.

# Generate the noisy signal
t_arr = np.arange(0, T, 1/fs)
x = np.array([A*np.sin(omega_0*t + phi) + np.random.normal(scale=sigma*np.sqrt(dt)) for t in t_arr])


# Compute the CRB for frequency
def CRB_f(t):
    CRB_approx = (6*sigma**2)/((A**2)*t**(2))
    fisher = (A**2)*(t*(2*t**2-3*t+1))/(12*sigma**2)
    CRB_better_approx = 1/((A**2/(6*sigma**2))*((t**(3))+((3*np.sin(2*t*omega_0)-6*t*omega_0*(t*omega_0*np.sin(2*t*omega_0)+np.cos(2*t*omega_0)))/(24*omega_0**3))))
    return CRB_approx

# Compute the CRB for frequency
def CRB_with_decay(t):
    fisher = (A**2/(2*sigma**2))*(np.exp(-2*(t-1)/T2)*(-np.exp(4/T2)*t**2+np.exp(2*t/T2)+np.exp(2*(t+1)/T2)+np.exp(2/T2)*(2*(t-1)-1)-(t-1)**2))/(np.exp(2/T2)-1)**3
    fisher_approx = (A**2/(2*sigma**2))*(((np.exp(-2/T2)+np.exp(-4/T2))/(1-np.exp(-2/T2))**3)-((np.exp(-2*t/T2)*t**2)/(1-np.exp(-2/T2))))
    CRB_approx = 1/fisher_approx
    return CRB_approx


# # Compute the error of frequency estimation over time
# f_est_array = np.zeros_like(t_arr)
# error_array = 100*np.ones_like(t_arr)
# CRB_f_array = np.zeros_like(t_arr)
# for i, ti in enumerate(t_arr):
#     if i > 100:
#         if is_ipFFT:
#             f_est_array[i] = ipFFT(x[0:i], fs=fs, fmax=2.0, win_type='hanning')
#         elif is_periodogram:
#             f, Pxx = signal.periodogram(x[0:i], fs=fs, window="hanning")
#             f_est_array[i] = 2*np.pi*f[np.argmax(Pxx)]
#         else:
#             raise ValueError("Unsupported frequency inference method.")
#
#         # save the errors and CRB
#         error_array[i] = np.abs(f_est_array[i] - omega_0)**2
#         CRB_f_array[i] = CRB_f(ti)
#
# error_array_smoothed = savitzky_golay(error_array, 1701, 3)
# error_array_np_avg = smooth_data_np_average(error_array, 71)
#
# # Plot the results
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
# ax1.plot(t_arr, f_est_array, label='Estimated frequency')
# ax1.axhline(y=omega_0, color='r', linestyle='--', label='True frequency')
# ax1.set_ylabel('Frequency (Hz)')
# ax1.set_title('Frequency Estimation over Time sigma=%r, omega=%r'%(sigma, omega_0))
# ax1.legend()
# # ax2.semilogy(t, error_array, label='Estimation Error')
# ax2.plot(t_arr, CRB_f_array, '--k', label='CRB')
# ax2.plot(t_arr, error_array_np_avg, label='smoothed')
# ax2.plot(t_arr, error_array_smoothed, label='golay')
# ax2.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Error (Hz)')
# ax2.set_title('Frequency Estimation Error over Time')
# ax2.legend()
# plt.show()

# CRB with exp

# Generate the noisy signal
t_arr = np.arange(0, T, 1/fs)
x = np.array([A*np.exp(-t/T2)*np.sin(omega_0*t + phi) + np.random.normal(scale=sigma*np.sqrt(dt)) for t in t_arr])

# Compute the error of frequency estimation over time
f_est_array = np.zeros_like(t_arr)
error_array = 100*np.ones_like(t_arr)
CRB_f_array = np.zeros_like(t_arr)
for i, ti in enumerate(t_arr):
    if i > 100:
        if is_ipFFT:
            f_est_array[i] = ipFFT(x[0:i], fs=fs, fmax=2.0, win_type='hanning')
        elif is_periodogram:
            f, Pxx = signal.periodogram(x[0:i], fs=fs, window="hanning")
            f_est_array[i] = 2*np.pi*f[np.argmax(Pxx)]
        else:
            raise ValueError("Unsupported frequency inference method.")

        # save the errors and CRB
        error_array[i] = np.abs(f_est_array[i] - omega_0)**2
        CRB_f_array[i] = CRB_with_decay(ti)


error_array_smoothed = savitzky_golay(error_array, 1701, 3)
error_array_np_avg = smooth_data_np_average(error_array, 71)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
ax1.plot(t_arr, f_est_array, label='Estimated frequency')
ax1.axhline(y=omega_0, color='r', linestyle='--', label='True frequency')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Frequency Estimation over Time sigma=%r, omega=%r'%(sigma, omega_0))
ax1.legend()
# ax2.semilogy(t, error_array, label='Estimation Error')
ax2.plot(t_arr, CRB_f_array, '--k', label='CRB')
ax2.plot(t_arr, error_array_np_avg, label='smoothed')
# ax2.plot(t_arr, error_array_smoothed, label='golay')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error (Hz)')
ax2.set_title('Frequency Estimation Error over Time')
ax2.legend()
plt.show()

