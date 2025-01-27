import numpy as np
import sdeint
import tqdm
import os
import matplotlib.pyplot as plt
def f_x(x, t):
    T2 = 0.87 * 1e-3  # Relaxation time
    w0 = 2 * np.pi * 1e4 * T2
    tau = 0.2
    f = np.zeros(3)
    f[0] += -x[0] + x[2] * x[1]
    f[1] += -x[2]*x[0] - x[1]
    # f[2] += (w0 - x[2]) / tau  # OU process
    f[2] += 0
    return f

def F(x, t):
    # Jacobian of f
    tau = 0.2
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

def G(x0, t):
    dc = 0.0
    tau = 0.0
    return np.diag([np.sqrt(2), np.sqrt(2), np.sqrt(dc)])  # Matrix G

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

def simulate(t_max, dt, x0, G, w0, tau, num=4):
    time_arr = np.arange(0, t_max, dt)
    # ALLOCATE MEMORY FOR THE ARRAYS=====================================================
    xs = np.array([np.zeros_like(x0) for _ in time_arr])
    x = x0
    for index, time in enumerate(tqdm.tqdm(time_arr, desc='pid:%r' % os.getpid())):
        # SIMULATION AND MEASUREMENT==============================
        x = step(x, time, dt, num)
        xs[index] = x
    return xs
