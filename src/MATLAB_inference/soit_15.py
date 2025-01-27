import numpy as np

def get_af(x, tau, w0):
    """
    Computes f(x), A (Jacobian), and b for the SDE.
    Args:
        x (numpy.ndarray): State vector.
        tau (float): Time constant of the OU process.
        w0 (float): Mean value of omega.
    Returns:
        f (numpy.ndarray): Drift vector.
        A (numpy.ndarray): Jacobian matrix.
        b (numpy.ndarray): Correction term (zeros here).
    """
    f = np.zeros(3)
    f[0] = -x[0] + x[2] * x[1]
    f[1] = -x[2] * x[0] - x[1]
    f[2] = (w0 - x[2]) / tau  # OU process

    A = np.array([
        [-1, x[2], x[1]],
        [-x[2], -1, -x[0]],
        [0, 0, -1 / tau]
    ])

    b = np.zeros(3)
    return f, A, b


def soit_15_01(x, h, nsteps, G, tau, w0):
    """
    Strong order 1.5 Ito-Taylor solver for SDE.
    Args:
        x (numpy.ndarray): Initial state vector.
        h (float): Time step.
        nsteps (int): Number of time steps.
        G (numpy.ndarray): Matrix G.
        tau (float): Time constant of the OU process.
        w0 (float): Mean value of the Larmour in non-dimensional units.
    Returns:
        xh (numpy.ndarray): Trajectory of states over time.
    """
    x = x.flatten()  # Ensure x is a 1D array
    nx = len(x)  # State dimension
    xh = np.zeros((nsteps, nx))  # Store states at each time step
    nw = G.shape[1]  # Number of Wiener processes
    nw_2 = 2 * nw

    sqh = np.sqrt(h)
    h2 = h * h / 2

    # Transformation matrix T
    T = np.block([
        [(np.sqrt(3) * sqh / 2) * np.eye(nw), (sqh / 2) * np.eye(nw)],
        [(h ** 1.5 / np.sqrt(3)) * np.eye(nw), np.zeros((nw, nw))]
    ])

    for k in range(nsteps):
        xh[k, :] = x  # Store current state

        # Generate random variables for the stochastic term
        r = T @ np.random.randn(nw_2)

        # Get f(x), A, and b
        f, A, b = get_af(x, tau, w0)

        # Update state using Ito-Taylor expansion
        x = x + h * f + h2 * (A @ f + b) + np.dot(np.hstack((G, A @ G)), r)

    return xh
