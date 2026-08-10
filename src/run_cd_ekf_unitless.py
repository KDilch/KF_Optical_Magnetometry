# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from space_state_model.unitless_magnetometer_model import UnitlessMagnetometerModel
from kalman_filter.unitless_cd_ekf import CD_EKF_unitless_magnetometer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simulation(configurator: UnitlessSimpleMagnetometerConfigurator, num_steps=20, save_path=None) -> dict:
    """
    Runs the SDE simulation step-by-step and extracts measurements.
    Saves simulation arrays to `save_path` (NPZ format) if specified.
    """
    sim_params = configurator.get_sim_params()
    h1 = configurator.h1
    t_max = configurator.tf / (configurator.T2 * 1e3)
    measure_every_nth = configurator.measure_every_nth
    sim_type = configurator.sim_type

    # Initialize the Simulation Model
    model = UnitlessMagnetometerModel(t=0.0, simulation_params=sim_params, logger=logger)
    
    # Create time grids
    time_arr = np.arange(0, t_max, h1)
    
    # Allocate memory for simulation results
    xs = np.zeros((len(time_arr), 3))
    z_s = np.zeros(len(time_arr))
    
    logger.info(f"Running unitless SDE simulation (type={sim_type}) for {len(time_arr)} steps...")
    
    # Run simulation step-by-step
    for index, time in enumerate(tqdm.tqdm(time_arr, desc='Simulation')):
        x, z = model.step(num_steps=num_steps)
        xs[index] = x
        z_s[index] = z
        
    # Generate measurements at every nth probing step
    yh = []
    t_meas = []
    x_sim_meas = []
    
    for idx, time in enumerate(time_arr):
        if idx % measure_every_nth == 0:
            yh.append(z_s[idx])
            t_meas.append(time)
            x_sim_meas.append(xs[idx])
            
    yh = np.array(yh)
    t_meas = np.array(t_meas)
    x_sim_meas = np.array(x_sim_meas)

    sim_data = {
        'time_arr': time_arr,
        'xs': xs,
        'z_s': z_s,
        't_meas': t_meas,
        'yh': yh,
        'x_sim_meas': x_sim_meas
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **sim_data)
        logger.info(f"Simulation data successfully saved to: {save_path}")

    return sim_data


def run_inference(configurator: UnitlessSimpleMagnetometerConfigurator, t_meas: np.ndarray, yh: np.ndarray, save_path=None) -> dict:
    """
    Runs EKF estimation over provided measurement arrays.
    Saves filter estimate arrays to `save_path` (NPZ) if specified.
    """
    filter_params = configurator.get_filter_params()
    ekf = CD_EKF_unitless_magnetometer(model_params=filter_params)
    
    # Allocate memory for EKF estimates
    x_est = np.zeros((len(yh), 3))
    P_est = np.zeros((len(yh), 3))
    
    logger.info(f"Running CD EKF updates and predictions for {len(t_meas)} measurement points...")
    
    # Run EKF filter
    for index, val in enumerate(tqdm.tqdm(t_meas, desc='EKF Filter')):
        ekf.update(yh[index])
        x_est[index] = ekf.x_est
        P_est[index] = [ekf.P_est[0, 0], ekf.P_est[1, 1], ekf.P_est[2, 2]]
        ekf.predict()

    inf_data = {
        'x_est': x_est,
        'P_est': P_est
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **inf_data)
        logger.info(f"Inference data successfully saved to: {save_path}")

    return inf_data


def generate_plots_and_report(
    configurator: UnitlessSimpleMagnetometerConfigurator,
    sim_data: dict,
    inf_data: dict,
    output_dir: str
):
    """
    Computes estimation errors, plots coordinates & errors with 3-sigma bounds,
    and generates a detailed Markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)
    T2 = configurator.T2
    xc = configurator.xc
    
    time_arr = sim_data['time_arr']
    xs = sim_data['xs']
    z_s = sim_data['z_s']
    t_meas = sim_data['t_meas']
    yh = sim_data['yh']
    x_sim_meas = sim_data['x_sim_meas']
    
    x_est = inf_data['x_est']
    P_est = inf_data['P_est']

    t2_ms = T2 * 1e3

    # --- Plot 1: Coordinates Plot (Sim vs EKF) ---
    fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(9, 9))
    
    # Jy plot
    axs[0].plot(time_arr * T2 * 1e3, xs[:, 0] / xc, label='Sim J_y', color='C0')
    axs[0].plot(t_meas * T2 * 1e3, x_est[:, 0] / xc, '--', label='EKF J_y', color='orange')
    axs[0].set_ylabel('J_y')
    axs[0].grid(True)
    
    # Jz plot
    axs[1].plot(time_arr * T2 * 1e3, xs[:, 1] / xc, label='Sim J_z', color='C0')
    axs[1].plot(t_meas * T2 * 1e3, x_est[:, 1] / xc, '--', label='EKF J_z', color='orange')
    axs[1].scatter(t_meas * T2 * 1e3, yh / xc, color='red', alpha=0.3, s=5, label='Meas (noisy)')
    axs[1].set_ylabel('J_z')
    axs[1].grid(True)
    
    # Frequency plot
    axs[2].plot(time_arr * T2 * 1e3, xs[:, 2] / T2, label='Sim omega', color='C0')
    axs[2].plot(t_meas * T2 * 1e3, x_est[:, 2] / T2, '--', label='EKF omega', color='orange')
    axs[2].set_ylabel('omega')
    axs[2].set_xlabel('Time (ms)')
    axs[2].grid(True)
    
    for ax in axs:
        ax.axvline(x=t2_ms, color='red', linestyle='--', label=f'T2 relaxation time ({t2_ms:.2f} ms)')
        ax.legend()
    
    plt.suptitle(f"Unitless Magnetometer EKF Tracking (type={configurator.sim_type})")
    coord_plot_path = os.path.join(output_dir, 'coordinates_plot.png')
    plt.savefig(coord_plot_path, dpi=150)
    plt.close()

    # --- Plot 2: Estimation Errors & 3-Sigma Bounds ---
    err = x_est - x_sim_meas
    sigma = np.sqrt(P_est)
    
    fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(9, 9))
    labels = ['J_y error', 'J_z error', 'omega error']
    scalings = [xc, xc, T2]
    
    for i in range(3):
        scaling = scalings[i]
        err_scaled = err[:, i] / scaling
        sigma_scaled = sigma[:, i] / scaling
        
        axs[i].plot(t_meas * T2 * 1e3, err_scaled, label='Error', color='purple')
        axs[i].fill_between(
            t_meas * T2 * 1e3, 
            -3 * sigma_scaled, 
            3 * sigma_scaled, 
            color='purple', 
            alpha=0.15, 
            label='+- 3-sigma covariance bounds'
        )
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        
    for ax in axs:
        ax.axvline(x=t2_ms, color='red', linestyle='--', label=f'T2 relaxation time ({t2_ms:.2f} ms)')
        ax.legend()
        
    axs[2].set_xlabel('Time (ms)')
    plt.suptitle(f"Estimation Errors and Covariance Bounds (type={configurator.sim_type})")
    error_plot_path = os.path.join(output_dir, 'errors_plot.png')
    plt.savefig(error_plot_path, dpi=150)
    plt.close()

    # --- Generate Markdown Report ---
    report_path = os.path.join(output_dir, 'report.md')
    abs_sim_path = os.path.abspath(os.path.join(output_dir, 'simulation_data.npz'))
    abs_inf_path = os.path.abspath(os.path.join(output_dir, 'inference_data.npz'))
    
    report_content = f"""# Unitless Magnetometer Simulation and EKF Run Report

**Date/Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Simulation Type:** {configurator.sim_type}

## Configuration parameters

### Physical & Solver Baseline parameters
- **Number of atoms (N):** {configurator.N}
- **Parameter q:** {configurator.q}
- **Relaxation time (T2):** {configurator.T2} s
- **Coupling constant (g_D):** {configurator.g_D} pA
- **Photon noise PSD (Sph):** {configurator.Sph} pA^2/Hz
- **Mean Larmor frequency (w0):** {configurator.w0} rad/s
- **Time step (h):** {configurator.h} s
- **Probing rate factor (measure_every_nth):** {configurator.measure_every_nth}
- **Diffusion constant (dc):** {configurator.dc}
- **Relaxation time constant (tau):** {configurator.tau}

### Derived unitless parameters
- **Effective atoms (Nq):** {configurator.Nq}
- **State scaling (xc):** {configurator.xc}
- **Measurement scaling (yc):** {configurator.yc}
- **Measurement noise (sig_v):** {configurator.sig_v}
- **Unitless step size (h1):** {configurator.h1}
- **Unitless measurement interval:** {configurator.meas_probing_rate_unitless}

## Data Files

- **Simulation Data:** [{os.path.basename(abs_sim_path)}](file:///{abs_sim_path.replace(os.sep, '/')})
- **EKF Inference Data:** [{os.path.basename(abs_inf_path)}](file:///{abs_inf_path.replace(os.sep, '/')})

## Visualizations

### Tracking Coordinates (Simulated vs Estimated)
![Coordinates Tracking](coordinates_plot.png)

### Estimation Errors with $\pm 3\sigma$ Covariance Bounds
![Estimation Errors](errors_plot.png)
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logger.info(f"Report and plots successfully generated in: {output_dir}")


def run_pipeline(
    configurator: UnitlessSimpleMagnetometerConfigurator = None,
    num_steps=20,
    data_input_path=None,
    output_dir=None
) -> str:
    """
    Main pipeline runner.
    If `data_input_path` is specified, SDE simulation is skipped and data is loaded from that file.
    Otherwise, SDE simulation runs.
    Runs EKF inference, generates coordinates & errors plots, and outputs report.md.
    """
    if configurator is None:
        configurator = UnitlessSimpleMagnetometerConfigurator()

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("runs", f"run_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    sim_data_path = os.path.join(output_dir, "simulation_data.npz")
    inf_data_path = os.path.join(output_dir, "inference_data.npz")

    # Step 1: Get Simulation Data (Load or Run)
    if data_input_path:
        logger.info(f"Loading simulation data from file: {data_input_path}")
        loaded = np.load(data_input_path)
        sim_data = {key: loaded[key] for key in loaded.files}
        # Copy loaded simulation data to target output folder for completeness
        np.savez(sim_data_path, **sim_data)
    else:
        sim_data = run_simulation(configurator, num_steps=num_steps, save_path=sim_data_path)

    # Step 2: Run CD EKF Inference
    inf_data = run_inference(
        configurator, 
        t_meas=sim_data['t_meas'], 
        yh=sim_data['yh'], 
        save_path=inf_data_path
    )

    # Step 3: Plots and Report Generation
    generate_plots_and_report(configurator, sim_data, inf_data, output_dir)
    
    return output_dir


if __name__ == "__main__":
    # Configure parameter mapping with final time tf equal to T2 (0.87 ms)
    config = UnitlessSimpleMagnetometerConfigurator(
        sim_type=None,
        tf=0.87,
        dc=0.0,
        tau=1e3,
        measure_every_nth=100
    )
    
    # Run the full pipeline (both SDE simulation and EKF inference)
    result_dir = run_pipeline(
        configurator=config, 
        num_steps=20,
        output_dir="runs/run_until_t2"
    )
    print(f"Pipeline executed successfully. View report at: {os.path.abspath(os.path.join(result_dir, 'report.md'))}")
