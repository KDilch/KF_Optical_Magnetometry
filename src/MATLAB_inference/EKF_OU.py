import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

def f_x(t, x, w0, tau):
    f = np.zeros(3)
    f[0] += -x[0] + x[2]*x[1]
    f[1] += -x[2]*x[0] - x[1]
    # f[2] += (w0 - x[2]) / tau  # OU process
    f[2] += 0
    return f

def F(x, t, tau):
    # Jacobian of f
    return np.array([
        [-1, x[2], x[1]],
        [-x[2], -1, -x[0]],
        [0, 0, 0]
        # [0, 0, -1 / tau]
    ])


def dP_dt(t, P, x, Q, dim_x, w0, tau):
    return np.reshape(np.dot(F(x, t, tau), np.reshape(P, (dim_x, dim_x))) +
                      np.dot(np.reshape(P, (dim_x, dim_x)), np.transpose(F(x, t, tau)) + Q),
                          dim_x ** 2)

def ekf_predict(t, x, P, Q, w0, tau, delta_t, dim_x):
    P_sol = solve_ivp(dP_dt,
                      [t, t + delta_t],
                      np.reshape(P, dim_x ** 2),
                      method='RK45',
                      dense_output=True,
                      # max_step=delta_t/1000,
                      args=(x,
                            Q,
                            dim_x,
                            w0,
                            tau))
    P_temp = P_sol.sol(t + delta_t)
    x_sol = solve_ivp(f_x,
                      [t, t + delta_t],
                      x,
                      method='RK45',
                      dense_output=True,
                      # max_step=delta_t/1000,
                      args=(w0, tau))
    x = x_sol.sol(t + delta_t)
    P = np.reshape(P_temp, (dim_x, dim_x))
    t += delta_t
    return x, P

def ekf_update(y, x, P, R_delta, dim_x):
    # y is like in the problem statement a measurement outcome
    H = np.array([[0, 1, 0]])
    innovation = np.array([y]) - np.dot(H, x)
    PHT = np.dot(P, H.T)
    S = np.dot(H, PHT) + R_delta
    S_INV = np.linalg.inv(S)
    K = np.dot(PHT, S_INV)
    # x = x + K*innovation
    x = x + np.dot(K, innovation)
    I_KH = np.identity(dim_x) - np.dot(K, H)
    P = np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(K, R_delta), K.T)
    return x, P

import numpy as np

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


def dx_dt_cc(t, x, K, y, w0, tau):
    return f_x(t, x, w0, tau) + np.dot(K, y)
def ekf_predict_update_cc(t, y, x, P, R_delta, delta_t, Q, w0, tau, dim_x):
    H = np.array([[0, 1, 0]])
    R_inv = np.linalg.inv(R_delta)
    K = np.dot(np.dot(P, H.T), R_inv)
    innovation = y - np.dot(H, x)
    P_sol = solve_ivp(dP_dt,
                      [t, t + delta_t],
                      np.reshape(P, dim_x ** 2),
                      method='RK45',
                      dense_output=True,
                      # max_step=delta_t/1000,
                      args=(x,
                            Q,
                            dim_x,
                            w0,
                            tau))
    P_temp = P_sol.sol(t + delta_t)
    x_sol = solve_ivp(dx_dt_cc,
                      [t, t + delta_t],
                      x,
                      method='RK45',
                      dense_output=True,
                      # max_step=delta_t/1000,
                      args=(K, innovation, w0, tau))
    x = x_sol.sol(t + delta_t)
    P = np.reshape(P_temp, (dim_x, dim_x))
    t += delta_t
    return x, P
