import data_transform
import numpy as np
import sdeint
import tqdm
import os
from scipy.integrate import solve_ivp


import matplotlib.pyplot as plt
import pandas as pd

# Parameters
N = 0.44 * 1e12  # Number of atoms
q = 0.198        # Parameter
T2 = 0.87 * 1e-3  # Relaxation time in s
g_D = 0.00177    # [pA]
Sph = 96         # Photon noise PSD [pA^2/Hz]
w0 = 2 * np.pi * 1e4  # Mean value of the Larmor in OU process
w01 = w0 * T2  # in non dimensional units
h = 50 * 1e-9    # Time step is the solver time step not the probing time!whath out
tf = 2.        # Final time in [ms]
nsteps = int(tf * 1e-3 / h)  # Number of steps in simulation
h1 = h / T2      # Time step in non-dimensional units
dc = 0.0         # Diffusion constant of the OU process 1 for fluctuating
tau = 0.2
measure_every_nth = 100
meas_probing_rate = h*measure_every_nth  # in s
meas_probing_rate_unitless = meas_probing_rate/T2
xc, yc, sig_v, T01 = data_transform.data_transform(T2, N, q, g_D, Sph, meas_probing_rate)  # state, measurement and time transformation
# Time constant of the OU process in non-dimensional units
G = np.diag([np.sqrt(2), np.sqrt(2), np.sqrt(dc)])  # Matrix G
D = np.array([[2, 0, 0], [0, 2, 0], [0, 0, dc]])

#sig_v = np.sqrt(2 * Sph / (g_D**2 * N * h))
xc2 = xc*xc
sgv2 = sig_v**2  # measurement noise variance
# Initial condition
x = np.array([0, np.sqrt(2 / (q * N)) * 0.5 * N, w0 * T2])

# Read the CSV file
# filename = "C:/Users/Klaudia/Documents/Python_projects/kalman_filters_in_magnetometry/src/MATLAB_inference/simulation_data_with_headers_dc0_short.csv"
# filename = "C:/Users/Klaudia/Documents/Python_projects/kalman_filters_in_magnetometry/src/MATLAB_inference/fluctuating_new_.csv"
# data = pd.read_csv(filename)


# Display the first few rows of the data
# print(data.head())

# SIMULATION
def f_x(x, t):
    f = np.zeros(3)
    f[0] += -x[0] + x[2] * x[1]
    f[1] += -x[2]*x[0] - x[1]
    # f[2] += (w0 - x[2]) / tau  # OU process
    f[2] += 0
    return f

def F(x, t):
    # Jacobian of f
    return np.array([
        [-1, x[2], x[1]],
        [-x[2], -1, -x[0]],
        [0, 0, 0]
        # [0, 0, -1 / tau]
    ])

def get_intrinsic_noise(dt):
    """Generates dW Wiener increment of the intrinsic noise."""
    return np.array([np.sqrt(dt),
                     np.sqrt(dt),
                     np.sqrt(dt)]) * np.random.randn(3)

def G(x, t):
    return np.diag([np.sqrt(2), np.sqrt(2), np.sqrt(dc)])    # Matrix G

def step(x0, t, dt, num_steps=20):
    t_span = np.linspace(t, t + dt, num_steps)
    dW = np.array([get_intrinsic_noise(dt) for _ in t_span[1:]])

    x = sdeint.itoSRI2(f=f_x,
                       G=G,
                       y0=x0,
                       tspan=t_span,
                       dW=dW)
    x = x[-1]
    return x

def simulate(t_max, dt, x0, num_steps=4):
    time_arr = np.arange(0, t_max, dt)
    # ALLOCATE MEMORY FOR THE ARRAYS=====================================================
    xs = np.array([np.zeros_like(x0) for _ in time_arr])
    x = x0
    for index, time in enumerate(tqdm.tqdm(time_arr, desc='pid:%r' % os.getpid())):
        # SIMULATION AND MEASUREMENT==============================
        x = step(x, time, dt, num_steps)
        xs[index] = x
    return xs


xs = simulate(tf/(T2*1e3), h1, x)

#Generate measurement outcomes
yh = []
for idx, x2_val in enumerate(xs[:, 1]):
    if idx % measure_every_nth == 0:  # Collect every nth sample
        yh.append(x2_val + sig_v * np.random.randn())  # Add noise to the measurement

#KALMAN HELPER FUNCTIONS
def dP_dt(t, P, x, Q, dim_x):
    return np.reshape(np.dot(F(x, t), np.reshape(P, (dim_x, dim_x))) +
                      np.dot(np.reshape(P, (dim_x, dim_x)), np.transpose(F(x, t)) + Q),
                          dim_x ** 2)

def f_x_ekf(t, x):
    return f_x(x, t)
def ekf_predict(t, x, P, Q, delta_t, dim_x):
    P_sol = solve_ivp(dP_dt,
                      [t, t + delta_t],
                      np.reshape(P, dim_x ** 2),
                      method='RK45',
                      dense_output=True,
                      # max_step=delta_t/1000,
                      args=(x,
                            Q,
                            dim_x))
    P_temp = P_sol.sol(t + delta_t)
    x_sol = solve_ivp(f_x_ekf,
                      [t, t + delta_t],
                      x,
                      method='RK45',
                      dense_output=True)
    x = x_sol.sol(t + delta_t)
    P = np.reshape(P_temp, (dim_x, dim_x))
    t += delta_t
    return x, P

def ekf_update_new(y, x, P, R_delta, dim_x):
    """
    Perform one EKF update step given the measurement y.

    Args:
        y: Scalar measurement value.
        x: Current state estimate (dim_x-dimensional vector).
        P: Current state covariance matrix (dim_x x dim_x).
        R_delta: Measurement noise covariance (scalar).
        dim_x: Dimensionality of the state.

    Returns:
        Updated state estimate (x) and covariance (P).
    """
    # Measurement matrix (maps state to measurement space)
    H = np.array([[0, 1, 0]])  # Shape: (1, dim_x)

    # Compute innovation (difference between actual and predicted measurement)
    innovation = np.array([y]) - np.dot(H, x)  # Shape: (1,)

    # Innovation covariance
    S = np.dot(H, np.dot(P, H.T)) + R_delta  # Shape: (1, 1)

    # Kalman gain
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))  # Shape: (dim_x, 1)

    # State update
    x = x + np.dot(K, innovation).flatten()  # Flatten to keep x as (dim_x,)

    # Covariance update (simplified form)
    P = P - np.dot(K, np.dot(S, K.T))

    return x, P
# # Kalman initialization
dw0 = 0.3
dm = 0.2
fr = w0 * (1 + dw0 * (2 * np.random.rand() - 1))
mr = 0.5 * xc * N * (1 + dm * (2 * np.random.rand() - 1))
m = np.array([0, mr, w0 * T2])
S = np.diag([
    (0.5 * xc * N * dm / 3)**2,
    (0.5 * xc * N * dm / 3)**2,
    (w0 * T2 * dw0 / 3)**2
])
m_ini = m / np.sqrt(2 / (q * N))  # True, not rescaled vals
S_ini = S / (2 / (q * N))

# Time vector in [ms] simulation
t = np.arange(0, len(xs[:, 2])) * h1
t_meas = np.arange(0, len(yh)) * meas_probing_rate_unitless
# Allocate memory
S_est = np.zeros((len(yh), 3))  # Equivalent to zeros(nsamples, 3) in MATLAB
x_est = np.zeros((len(yh), 3))

for index, val in enumerate(t_meas):
    # m, S = ekf_update_new(yh[index], m, S, R_delta=sgv2, dim_x=3)
    x_est[index, :] = m.T  # Store the state estimate (transpose if m is a row vector)
    S_est[index, :] = [S[0, 0], S[1, 1], S[2, 2]]
    m, S = ekf_predict(index*meas_probing_rate_unitless, m, S, D, delta_t=meas_probing_rate_unitless, dim_x=3)

    # print(m, x1[index], x2[index], x3[index], "estimate and true value")
# Larmor frequency
f_L = xs[:, 2] / (2 * np.pi * T2 * 1e3)  # [kHz]
f_ud = (w0 * T2 + 3 * np.sqrt(dc) * np.array([-1, 1])) / (2 * np.pi * T2 * 1e3)  # 3σ bounds
#
# Plot Larmor frequency
plt.plot(t_meas, x_est[:, 1], label='EKF')
plt.plot(t, xs[:, 1], label='Larmor Frequency')
# plt.axhline(f_ud[0], linestyle='--', color='red', label='Lower Bound')
# plt.axhline(f_ud[1], linestyle='--', color='green', label='Upper Bound')
plt.xlabel('t [ms]')
plt.ylabel('f_L [kHz]')
plt.title('Larmor frequency, OU process')
plt.legend()
plt.show()
#
plt.plot(t_meas, x_est[:, 2])
plt.plot(t, xs[:, 2])
plt.show()
#
# # Plot Larmor frequency
# plt.plot(t, x2, label='Jz')
# plt.plot(t_meas, x_est[:, 1], label='EKF')
# # plt.axhline(f_ud[0], linestyle='--', color='red', label='Lower Bound')
# # plt.axhline(f_ud[1], linestyle='--', color='green', label='Upper Bound')
# plt.xlabel('t [ms]')
# plt.ylabel('Jz')
# plt.title('JZ')
# plt.legend()
# plt.show()


# # Access specific columns
# time = data['Time (ms)']
# y_t = data['y(t)']
# x1 = data['x1']*xc
# x2 = data['x2']*xc
# x3 = data['x3']*(2*np.pi*T2*1e3)
#
#
# # True measurement outcomes
# yh = []  # Use a list for efficient appending
# for idx, x2_val in enumerate(x2):
#     if idx % measure_every_nth == 0:  # Collect every nth sample
#         yh.append(x2_val + sig_v * np.random.randn())  # Add noise to the measurement
#
# # Convert to NumPy array after appending
# yh = np.array(yh)
#
# # Kalman initialization
# dw0 = 0.1
# dm = 0.2
# fr = w0 * (1 + dw0 * (2 * np.random.rand() - 1))
# mr = 0.5 * xc * N * (1 + dm * (2 * np.random.rand() - 1))
# m = np.array([0, mr, w0 * T2])
# S = np.diag([
#     (0.5 * xc * N * dm / 3)**2,
#     (0.5 * xc * N * dm / 3)**2,
#     (w0 * T2 * dw0 / 3)**2
# ])
# m_ini = m / np.sqrt(2 / (q * N))  # True, not rescaled vals
# S_ini = S / (2 / (q * N))
#
# # Time vector in [ms] simulation
# t = 1e3 * np.arange(0, nsteps) * h
# t_meas = 1e3 * np.arange(0, len(yh))*meas_probing_rate
# # Allocate memory
# S_est = np.zeros((len(yh), 3))  # Equivalent to zeros(nsamples, 3) in MATLAB
# x_est = np.zeros((len(yh), 3))
#
# for index, val in enumerate(t_meas):
#     m, S = ekf_update(yh[index], m, S, R_delta=np.array([[sgv2]]), dim_x=3)
#     x_est[index, :] = m.T  # Store the state estimate (transpose if m is a row vector)
#     S_est[index, :] = [S[0, 0], S[1, 1], S[2, 2]]
#     m, S = ekf_predict(index*T01, m, S, D, w01, tau, delta_t=T01, dim_x=3)
#
#     # print(m, x1[index], x2[index], x3[index], "estimate and true value")
# # Larmor frequency
# f_L = x3 / (2 * np.pi * T2 * 1e3)  # [kHz]
# f_ud = (w0 * T2 + 3 * np.sqrt(dc) * np.array([-1, 1])) / (2 * np.pi * T2 * 1e3)  # 3σ bounds
#
# # Plot Larmor frequency
# plt.plot(t, x3, label='Larmor Frequency')
# plt.plot(t_meas, x_est[:, 2], label='EKF')
# # plt.axhline(f_ud[0], linestyle='--', color='red', label='Lower Bound')
# # plt.axhline(f_ud[1], linestyle='--', color='green', label='Upper Bound')
# plt.xlabel('t [ms]')
# plt.ylabel('f_L [kHz]')
# plt.title('Larmor frequency, OU process')
# plt.legend()
# plt.show()
#
# plt.plot(t_meas, x_est[:, 2]/(2*np.pi*1e3*T2**2*w0))
# plt.show()
#
# # Plot Larmor frequency
# plt.plot(t, x2, label='Jz')
# plt.plot(t_meas, x_est[:, 1], label='EKF')
# # plt.axhline(f_ud[0], linestyle='--', color='red', label='Lower Bound')
# # plt.axhline(f_ud[1], linestyle='--', color='green', label='Upper Bound')
# plt.xlabel('t [ms]')
# plt.ylabel('Jz')
# plt.title('JZ')
# plt.legend()
# plt.show()
#
# #Save data to file
# # Create headers for the CSV
# headers = ['x_est_1', 'x_est_2', 'x_est_3', 'S_est_1', 'S_est_2', 'S_est_3']
#
# # Concatenate x_est and S_est along columns
# data_to_save = np.hstack((x_est, S_est))
#
# # Create a DataFrame for saving
# df = pd.DataFrame(data_to_save, columns=headers)
#
# # Save to a CSV file
# filename = 'estimates_data.csv'
# df.to_csv(filename, index=False)
#
# print(f"Data saved to {filename}")
#
# # # Spin J_z
# # xi = x2 * np.sqrt(q * N / 2)
# # xi_ud = 3 * np.array([-1, 1]) * np.sqrt(q * N / 2)
#
# # # Plot Spin J_z
# # plt.subplot(212)
# # plt.plot(t, xi, label='J_z')
# # # plt.axhline(xi_ud[0], linestyle='--', color='red', label='Lower Bound')
# # # plt.axhline(xi_ud[1], linestyle='--', color='green', label='Upper Bound')
# # plt.xlabel('t [ms]')
# # plt.ylabel('Spin J_z')
# # plt.title('J_z, spin component')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
#
# # # Save the data to a CSV file
# # output_data = np.column_stack([
# #     t, yh, x1 * np.sqrt(q * N / 2), x2 * np.sqrt(q * N / 2), f_L
# # ])
# # headers = ['Time (ms)', 'y(t)', 'x1', 'x2', 'x3']
# #
# # # Use pandas to write the data
# # df = pd.DataFrame(output_data, columns=headers)
# # df.to_csv('simulation_data_with_headers.csv', index=False)
#
# # print('CSV file with headers created successfully.')