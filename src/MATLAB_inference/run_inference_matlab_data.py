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
# g_D = 0.1
Sph = 96         # Photon noise PSD [pA^2/Hz]
w0 = 2 * np.pi * 1e4  # Mean value of the Larmor in OU process
w01 = w0 * T2  # in non dimensional units
h = 50 * 1e-9    # Time step is the solver time step not the probing time!whath out
tf = 2.        # Final time in [ms]
nsteps = int(tf * 1e-3 / h)  # Number of steps in simulation
h1 = h / T2      # Time step in non-dimensional units
dc = 0.0
# dc = 0.01         # Diffusion constant of the OU process 0.01 for fluctuating
tau = 0.001
measure_every_nth = 1000
meas_probing_rate = h*measure_every_nth  # in s
meas_probing_rate_unitless = meas_probing_rate/T2
xc, yc, sig_v, T01 = data_transform.data_transform(T2, N, q, g_D, Sph, meas_probing_rate)  # state, measurement and time transformation
# Time constant of the OU process in non-dimensional units
G = np.diag([np.sqrt(2), np.sqrt(2), np.sqrt(dc)])  # Matrix G
D = np.array([[2, 0, 0], [0, 2, 0], [0, 0, dc]])

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
    # f[2] += (w01 - x[2]) / tau  # OU process
    f[2] += 0
    return f

def get_intrinsic_noise(dt):
    """Generates dW Wiener increment of the intrinsic noise."""
    return np.array([np.sqrt(dt),
                     np.sqrt(dt),
                     np.sqrt(dt)]) * np.random.randn(3)

def get_G(x, t):
    return G
def step(x0, t, dt, num_steps=20):
    t_span = np.linspace(t, t + dt, num_steps)
    dW = np.array([get_intrinsic_noise(dt) for _ in t_span[1:]])

    x = sdeint.itoSRI2(f=f_x,
                       G=get_G,
                       y0=x0,
                       tspan=t_span,
                       dW=dW)
    return x[-1]

def simulate(t_max, dt, x0, xs, num_steps=20):
    time_arr = np.arange(0, t_max, dt)
    for index, time in enumerate(tqdm.tqdm(time_arr, desc='pid:%r' % os.getpid())):
        # SIMULATION AND MEASUREMENT==============================
        x = step(x0, time, dt, num_steps)
        xs[index] = x
        x0 = x
    return xs


time_arr_sim_unitless = np.arange(0, tf/(T2*1e3), h1)
xs = np.array([np.zeros_like(x) for _ in time_arr_sim_unitless])
simulate(tf/(T2*1e3), h1, x, xs)

#Generate measurement outcomes
yh = []
for idx, x1_val in enumerate(xs[:, 1]):
    if idx % measure_every_nth == 0:  # Collect every nth sample
        yh.append(x1_val + sig_v * np.random.randn())  # Add noise to the measurement
#KALMAN HELPER FUNCTIONS
def dP_dt(t, P, x, Q, dim_x):
    return np.reshape(np.dot(F(x, t), np.reshape(P, (dim_x, dim_x))) +
                      np.dot(np.reshape(P, (dim_x, dim_x)), np.transpose(F(x, t)) + Q),
                          dim_x ** 2)

def F(x, t):
    # Jacobian of f
    return np.array([
        [-1, x[2], x[1]],
        [-x[2], -1, -x[0]],
        [0, 0, 0]
        # [0, 0, -1 / tau]
    ])
def f_x_ekf(t, x):
    f = np.zeros(3)
    f[0] += -x[0] + x[2] * x[1]
    f[1] += -x[2] * x[0] - x[1]
    # f[2] += (w01 - x[2]) / tau  # OU process
    f[2] += 0
    return f
def ekf_predict(t, x, P, Q, delta_t, dim_x):
    P_sol = solve_ivp(dP_dt,
                      [t, t + delta_t],
                      np.reshape(P, dim_x ** 2),
                      method='RK45',
                      dense_output=True,
                      max_step=delta_t/(measure_every_nth),
                      args=(x,
                            Q,
                            dim_x))
    P_temp = P_sol.sol(t + delta_t)
    x_sol = solve_ivp(f_x_ekf,
                      [t, t + delta_t],
                      x,
                      method='RK45',
                      max_step=delta_t/(measure_every_nth),
                      dense_output=True)
    x = x_sol.sol(t + delta_t)
    P = np.reshape(P_temp, (dim_x, dim_x))
    # P = 0.5 * (P + P.T)
    return x, P


def ekf_update_new(y, x, P, R_delta):
    """
    Perform one EKF update step given the measurement y.

    Args:
        y: Scalar measurement value.
        x: Current state estimate (dim_x-dimensional vector).
        P: Current state covariance matrix (dim_x x dim_x).
        R_delta: Measurement noise covariance (scalar).

    Returns:
        Updated state estimate (x) and covariance (P).
    """
    H = np.array([[0., 1., 0.]])  # Shape: (1, dim_x)

    # Compute innovation (difference between actual and predicted measurement)
    innovation = np.array([y]) - np.dot(H, x)  # Shape: (1,)

    # Innovation covariance
    S = np.dot(H, np.dot(P, H.T)) + R_delta  # Shape: (1, 1)

    # Kalman gain
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))  # Shape: (dim_x, 1)

    # State update
    x_new = x + np.dot(K, innovation)

    P_new = (np.eye(3)-K@H)@P
    P_new = 0.5*(P_new+P_new.T)

    return x_new, P_new


# Kalman initialization
dw0 = 0.01
dJ = 0.01
x_current = np.array([0, 0.5 * xc * N * (1 + dJ), w0 * (1 + dw0)*T2])
P_current = np.diag([
    (0.5 * xc * N * dJ/3)**2,
    (0.5 * xc * N * dJ/3)**2,
    (w0 * T2 * dw0/3)**2
])

# Time vector in [ms] simulation
t = np.arange(0, len(xs[:, 2])) * h1
t_meas = np.arange(0, len(yh)) * meas_probing_rate_unitless

# Allocate memory
P_est = np.zeros((len(yh), 3))
x_est = np.zeros((len(yh), 3))
print(x_current, xs[0, :], "initial estimate")

for index, val in enumerate(tqdm.tqdm(t_meas, desc='pid:%r' % os.getpid())):
    x_current, P_current = ekf_update_new(yh[index], x_current, P_current, R_delta=sgv2)
    x_est[index, :] = x_current.T  # Store the state estimate (transpose if m is a row vector)
    P_est[index, :] = [P_current[0, 0], P_current[1, 1], P_current[2, 2]]
    x_current, P_current = ekf_predict(val, x_current, P_current, D, delta_t=meas_probing_rate_unitless, dim_x=3)

    # print(m, x1[index], x2[index], x3[index], "estimate and true value")
# Larmor frequency
f_L = xs[:, 2] / (2 * np.pi * T2 * 1e3)  # [kHz]
f_ud = (w0 * T2 + 3 * np.sqrt(dc) * np.array([-1, 1])) / (2 * np.pi * T2 * 1e3)  # 3σ bounds
#
# Plot Larmor frequency
plt.plot(t_meas, x_est[:, 1], label='EKF')
plt.plot(t, xs[:, 1], label='Jz')
# plt.axhline(f_ud[0], linestyle='--', color='red', label='Lower Bound')
# plt.axhline(f_ud[1], linestyle='--', color='green', label='Upper Bound')
plt.xlabel('t [ms]')
plt.ylabel('J_z')
plt.title('Jz, OU process')
plt.legend()
plt.show()
#
plt.plot(t_meas, x_est[:, 0])
plt.plot(t, xs[:, 0])
plt.title('J_y')
plt.legend()
plt.show()

plt.plot(t_meas, x_est[:, 2])
plt.plot(t, xs[:, 2])
plt.show()
