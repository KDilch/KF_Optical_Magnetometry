# -*- coding: utf-8 -*-
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
import multiprocessing.pool
import scipy.io as sio

from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from kalman_filter.unitless_cd_ekf import CD_EKF_unitless_magnetometer
from kalman_filter.unitless_cd_ckf import CD_CKF_unitless_magnetometer


def generate_matlab_data(nsamples, f_L, T2, g_D, Sph, q, N, T0):
    """
    Generates unitless synthetic data matching MATLAB's get_output_data.m exactly.
    """
    om_L = 2 * np.pi * f_L
    th0 = om_L * T2
    
    # State scaling and measurement transformation
    Nq = N * q
    xc = np.sqrt(2 / Nq)
    sig_v = np.sqrt(2 * Sph / (g_D**2 * Nq * T0))
    T01 = T0 / T2
    
    # Initial state
    x = np.array([0.0, 0.5 * xc * N])
    
    c = np.cos(th0 * T01)
    s = np.sin(th0 * T01)
    A = np.exp(-T01) * np.array([[c, s], [-s, c]])
    G = np.sqrt(1.0 - np.exp(-2.0 * T01)) * np.eye(2)
    
    y = np.zeros(nsamples)
    x_sim = np.zeros((nsamples, 3))
    
    for k in range(nsamples):
        y[k] = x[1] + sig_v * np.random.randn()
        x_sim[k, :2] = x
        x_sim[k, 2] = th0
        
        # Propagate spins
        x = A @ x + G @ np.random.randn(2)
        
    return y, x_sim


def run_filter(filter_cls, configurator, y, T2, x0_filter, P0_filter):
    """
    Runs a single filter trial with customized initial state/covariance.
    """
    filter_params = configurator.get_filter_params()
    if filter_cls == CD_CKF_unitless_magnetometer:
        filter_params.inference_method = 'discrete'
        
    filter_params.x_0 = x0_filter.copy()
    filter_params.P0 = P0_filter.copy()
    
    filt = filter_cls(model_params=filter_params)
    
    # Run loop
    for k in range(len(y)):
        filt.update(y[k])
        filt.predict()
        
    # Scale final frequency estimate back to physical units (Hz)
    cf = 1.0 / (2.0 * np.pi * T2)
    final_freq_hz = filt.x_est[2] * cf
    return final_freq_hz


def run_single_trial(trial_idx, config, f_L, T2, g_D, Sph, q, N, T0, nsamples):
    """
    Worker function to run a single EKF and CKF trial.
    Must be defined at module top-level for multiprocessing picklability.
    """
    # Set random seed per trial for repeatability
    np.random.seed(trial_idx)
    
    # Generate simulated data matching Matlab
    y, _ = generate_matlab_data(nsamples, f_L, T2, g_D, Sph, q, N, T0)
    
    # Set seed for random initialization
    np.random.seed(trial_idx + 1000)
    df_L = 0.2
    dm = 0.2
    fr = f_L * (1.0 + df_L * (2.0 * np.random.rand() - 1.0))
    mr = 0.5 * config.xc * N * (1.0 + dm * (2.0 * np.random.rand() - 1.0))
    
    x0_filter = np.array([0.0, mr, 2.0 * np.pi * fr * T2])
    P0_filter = np.diag([
        (0.5 * config.xc * N * dm / 3.0) ** 2,
        (0.5 * config.xc * N * dm / 3.0) ** 2,
        (2.0 * np.pi * f_L * T2 * df_L / 3.0) ** 2
    ])
    
    # Run EKF
    ekf_val = run_filter(CD_EKF_unitless_magnetometer, config, y, T2, x0_filter, P0_filter)
    ekf_err_mhz = 1e3 * (ekf_val - f_L)
    
    # Run CKF
    ckf_val = run_filter(CD_CKF_unitless_magnetometer, config, y, T2, x0_filter, P0_filter)
    ckf_err_mhz = 1e3 * (ckf_val - f_L)
    
    return ekf_err_mhz**2, ckf_err_mhz**2


def run_benchmark():
    print("=== Starting EKF vs CKF Benchmark over varying N (Parallelized, 1000 Simulations) ===")
    
    # Load CRB and PEM values from Matlab folder
    mat1 = sio.loadmat("src/Matlab_simulations_KD/crb_vs_natoms_1000.mat")
    mat2 = sio.loadmat("src/Matlab_simulations_KD/pem_er_na_1000.mat")
    
    na = mat1["na"].flatten()
    crb_vals = mat1["crb"].flatten()
    pem_vals = mat2["mer"].flatten()
    
    # Use exact 30 N-values from Matlab for comparison
    N_values = na
    
    # Physical parameters matching Matlab
    f_L = 1e4            # Larmor frequency [Hz]
    f0 = 600.0           # [Hz]
    f1 = 550.0           # [Hz]
    g_D = 0.00177        # [pA]
    Sph = 96.0           # [pA^2/Hz]
    q = 0.198
    T0 = 5.0 * 1e-6      # Sampling time [s]
    
    nsamples = 1000      # Length of simulation
    ntries = 10          # Number of trials for averaging (At least 1000!)
    
    ekf_rmses = []
    ckf_rmses = []
    as_crbs = []
    
    # Main loop over N
    for idx, N in enumerate(N_values):
        T2 = 1.0 / (f0 + f1 * 1e-12 * N)
        cf = 1.0 / (2.0 * np.pi * T2)
        
        # Calculate Asymptotic CRB (converted to mHz)
        th0 = 2.0 * np.pi * T2 * f_L
        as_crb = 1e3 * np.sqrt(8.0 * Sph) / (np.pi * T2**1.5 * g_D * N)
        as_crb = as_crb * (th0**2 + 1.0)**1.5 / (th0 * np.sqrt(th0**4 + 3.0 * th0**2 + 6.0))
        as_crbs.append(as_crb)
        
        config = UnitlessSimpleMagnetometerConfigurator(
            N=N,
            q=q,
            T2=T2,
            g_D=g_D,
            Sph=Sph,
            w0=2.0 * np.pi * f_L,
            h=T0,
            tf=nsamples * T0 * 1e3,
            dc=0.0,
            tau=1e3,
            measure_every_nth=1,
            sim_type=None
        )
        
        # Run ntries parallelized trials using standard starmap
        print(f"Running N = 10^{np.log10(N):.1f} ({idx+1}/{len(N_values)}) over {ntries} parallelized trials...")
        pool = multiprocessing.Pool(processes=max(1, os.cpu_count() - 1))
        args = [(trial, config, f_L, T2, g_D, Sph, q, N, T0, nsamples) for trial in range(ntries)]
        
        results = pool.starmap(run_single_trial, args)
            
        pool.close()
        pool.join()
        
        ekf_err_sq_sum = sum(r[0] for r in results)
        ckf_err_sq_sum = sum(r[1] for r in results)
        
        ekf_rmse = np.sqrt(ekf_err_sq_sum / ntries)
        ckf_rmse = np.sqrt(ckf_err_sq_sum / ntries)
        
        ekf_rmses.append(ekf_rmse)
        ckf_rmses.append(ckf_rmse)
        
        print(f"  N = 10^{np.log10(N):.1f}: EKF RMSE = {ekf_rmse:.3f} mHz | CKF RMSE = {ckf_rmse:.3f} mHz | CRB = {crb_vals[idx]:.3f} mHz")
        
    ekf_rmses = np.array(ekf_rmses)
    ckf_rmses = np.array(ckf_rmses)
    as_crbs = np.array(as_crbs)
    
    # Save the plot
    os.makedirs("runs", exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.loglog(N_values, ekf_rmses, 'o-', linewidth=2, label='EKF (Python)', color='orange')
    plt.loglog(N_values, ckf_rmses, 's-', linewidth=2, label='CKF (Python)', color='green')
    plt.loglog(N_values, as_crbs, '--', linewidth=2, label='Asymptotic CRB', color='purple')
    plt.loglog(na, crb_vals, 'k*-', linewidth=1.5, label='CRB (Matlab)')
    plt.loglog(na, pem_vals, 'v-', linewidth=1.5, label='PEM (Matlab)', color='blue')
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Number of atoms N')
    plt.ylabel('Frequency estimation error [mHz]')
    plt.title('Frequency Estimation Error (RMSE) vs Number of Atoms N (1000 Trials)')
    plt.legend()
    plot_path = "runs/ekf_vs_ckf_vs_crb_N.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Benchmark plot successfully saved to: {os.path.abspath(plot_path)}")
    
    # Save results to markdown file
    results_path = "runs/benchmark_results.md"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("# EKF vs CKF Performance Comparison over varying N (1000 Trials)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Simulation length:** {nsamples} steps\n")
        f.write(f"**Averaging:** {ntries} trials\n\n")
        f.write("| Number of Atoms N | EKF RMSE (mHz) | CKF RMSE (mHz) | Asympt. CRB (mHz) | CRB (mHz) | PEM (mHz) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for i in range(len(N_values)):
            f.write(f"| 10^{np.log10(N_values[i]):.1f} | {ekf_rmses[i]:.3f} | {ckf_rmses[i]:.3f} | {as_crbs[i]:.3f} | {crb_vals[i]:.3f} | {pem_vals[i]:.3f} |\n")
        f.write(f"\nComparative Loglog Plot: [ekf_vs_ckf_vs_crb_N.png](ekf_vs_ckf_vs_crb_N.png)\n")
    print(f"Benchmark results report successfully saved to: {os.path.abspath(results_path)}")


if __name__ == '__main__':
    # On Windows, freeze_support is required for multiprocessing packaging
    multiprocessing.freeze_support()
    run_benchmark()
