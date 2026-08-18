# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from space_state_model.unitless_magnetometer_model import UnitlessMagnetometerModel
from kalman_filter.unitless_cd_ekf import CD_EKF_unitless_magnetometer
from kalman_filter.unitless_cd_ckf import CD_CKF_unitless_magnetometer
from evaluation.crb_pem import PredictionErrorMethod, CramerRaoBounds

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
    
    # Pre-allocate lists for measurements
    yh = []
    t_meas = []
    x_sim_meas = []
    
    logger.info(f"Running unitless SDE simulation (type={sim_type}) for {len(time_arr)} steps...")
    
    # Run simulation step-by-step
    for index, time in enumerate(tqdm.tqdm(time_arr, desc='Simulation')):
        x, _ = model.step(num_steps=num_steps)
        xs[index] = x
        
        # Read the sensor with noise only at measurement intervals
        if index % measure_every_nth == 0:
            yh.append(model.read_sensor(with_noise=True))
            t_meas.append(time)
            x_sim_meas.append(x)
            
    yh = np.array(yh)
    t_meas = np.array(t_meas)
    x_sim_meas = np.array(x_sim_meas)
 
    sim_data = {
        'time_arr': time_arr,
        'xs': xs,
        'z_s': xs[:, 1],  # Store clean J_z trajectory as baseline z_s
        't_meas': t_meas,
        'yh': yh,
        'x_sim_meas': x_sim_meas
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **sim_data)
        logger.info(f"Simulation data successfully saved to: {save_path}")

    return sim_data


def run_inference(configurator: UnitlessSimpleMagnetometerConfigurator, t_meas: np.ndarray, yh: np.ndarray, save_path=None, filter_cls=None, desc='Filter') -> dict:
    """
    Runs estimation over provided measurement arrays using the specified filter class.
    Saves filter estimate arrays to `save_path` (NPZ) if specified.
    """
    if filter_cls is None:
        filter_cls = CD_EKF_unitless_magnetometer
        
    filter_params = configurator.get_filter_params()
    # For CKF, we default the inference method to discrete (analytical state transition)
    if filter_cls == CD_CKF_unitless_magnetometer:
        filter_params.inference_method = 'discrete'
        
    filt = filter_cls(model_params=filter_params)
    
    # Allocate memory for estimates
    x_est = np.zeros((len(yh), 3))
    P_est = np.zeros((len(yh), 3))
    
    logger.info(f"Running CD {desc} updates and predictions for {len(t_meas)} measurement points...")
    
    # Run filter
    for index, val in enumerate(tqdm.tqdm(t_meas, desc=desc)):
        filt.update(yh[index])
        x_est[index] = filt.x_est
        P_est[index] = [filt.P_est[0, 0], filt.P_est[1, 1], filt.P_est[2, 2]]
        filt.predict()

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
    ekf_data: dict,
    ckf_data: dict,
    output_dir: str,
    timestamp: str = None,
    run_time: datetime = None,
    crb_data: dict = None
):
    """
    Computes estimation errors, plots coordinates & errors with 3-sigma bounds for EKF and CKF,
    and generates a detailed comparison report.
    """
    os.makedirs(output_dir, exist_ok=True)
    if run_time is None:
        run_time = datetime.now()
    if timestamp is None:
        timestamp = run_time.strftime("%Y%m%d_%H%M%S")

    T2 = configurator.T2
    xc = configurator.xc
    
    time_arr = sim_data['time_arr']
    xs = sim_data['xs']
    t_meas = sim_data['t_meas']
    yh = sim_data['yh']
    x_sim_meas = sim_data['x_sim_meas']
    
    x_ekf = ekf_data['x_est']
    P_ekf = ekf_data['P_est']
    
    x_ckf = ckf_data['x_est']
    P_ckf = ckf_data['P_est']

    t2_ms = T2 * 1e3

    # --- Plot 1: Coordinates Plot (Sim vs EKF vs CKF) ---
    fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(9, 9))
    
    # Jy plot
    axs[0].plot(time_arr * T2 * 1e3, xs[:, 0] / xc, label='Sim J_y', color='C0')
    axs[0].plot(t_meas * T2 * 1e3, x_ekf[:, 0] / xc, '--', label='EKF J_y', color='orange')
    axs[0].plot(t_meas * T2 * 1e3, x_ckf[:, 0] / xc, ':', label='CKF J_y', color='green')
    axs[0].set_ylabel('J_y')
    axs[0].grid(True)
    
    # Jz plot
    axs[1].plot(time_arr * T2 * 1e3, xs[:, 1] / xc, label='Sim J_z', color='C0')
    axs[1].plot(t_meas * T2 * 1e3, x_ekf[:, 1] / xc, '--', label='EKF J_z', color='orange')
    axs[1].plot(t_meas * T2 * 1e3, x_ckf[:, 1] / xc, ':', label='CKF J_z', color='green')
    axs[1].scatter(t_meas * T2 * 1e3, yh / xc, color='red', alpha=0.3, s=5, label='Meas (noisy)')
    axs[1].set_ylabel('J_z')
    axs[1].grid(True)
    
    # Frequency plot
    axs[2].plot(time_arr * T2 * 1e3, xs[:, 2] / T2, label='Sim omega', color='C0')
    axs[2].plot(t_meas * T2 * 1e3, x_ekf[:, 2] / T2, '--', label='EKF omega', color='orange')
    axs[2].plot(t_meas * T2 * 1e3, x_ckf[:, 2] / T2, ':', label='CKF omega', color='green')
    axs[2].set_ylabel('omega')
    axs[2].set_xlabel('Time (ms)')
    axs[2].grid(True)
    
    for ax in axs:
        ax.axvline(x=t2_ms, color='red', linestyle='--', label=f'T2 relaxation time ({t2_ms:.2f} ms)')
        ax.legend()
    
    plt.suptitle(f"Unitless Magnetometer EKF vs CKF Comparison (type={configurator.sim_type})")
    coord_plot_filename = f"coordinates_plot_{timestamp}.png"
    coord_plot_path = os.path.join(output_dir, coord_plot_filename)
    plt.savefig(coord_plot_path, dpi=150)
    plt.close()

    # --- Plot 2: Estimation Errors & 3-Sigma Bounds (Omega Only) ---
    err_ekf = x_ekf - x_sim_meas
    sigma_ekf = np.sqrt(P_ekf)
    
    err_ckf = x_ckf - x_sim_meas
    sigma_ckf = np.sqrt(P_ckf)
    
    fig_err, ax_e = plt.subplots(layout='constrained', figsize=(9, 5))
    
    err_ekf_scaled = err_ekf[:, 2] / T2
    sigma_ekf_scaled = sigma_ekf[:, 2] / T2
    
    err_ckf_scaled = err_ckf[:, 2] / T2
    sigma_ckf_scaled = sigma_ckf[:, 2] / T2
    
    ax_e.plot(t_meas * T2 * 1e3, err_ekf_scaled, label='EKF Error', color='orange')
    ax_e.fill_between(
        t_meas * T2 * 1e3, 
        -3 * sigma_ekf_scaled, 
        3 * sigma_ekf_scaled, 
        color='orange', 
        alpha=0.1, 
        label='EKF +- 3-sigma bounds'
    )
    
    ax_e.plot(t_meas * T2 * 1e3, err_ckf_scaled, label='CKF Error', color='green')
    ax_e.fill_between(
        t_meas * T2 * 1e3, 
        -3 * sigma_ckf_scaled, 
        3 * sigma_ckf_scaled, 
        color='green', 
        alpha=0.1, 
        label='CKF +- 3-sigma bounds'
    )
    
    # Plot steady-state Asymptotic CRB bounds if available
    if crb_data is not None:
        asympt_crb_mhz = crb_data['asympt_crb']
        crb_std_rad_s = (asympt_crb_mhz * 1e-3) * 2.0 * np.pi
        ax_e.axhline(y=3.0 * crb_std_rad_s, color='blue', linestyle=':', label='Asympt. CRB +3-sigma limit')
        ax_e.axhline(y=-3.0 * crb_std_rad_s, color='blue', linestyle=':', label='Asympt. CRB -3-sigma limit')
        
    ax_e.axvline(x=t2_ms, color='red', linestyle='--', label=f'T2 relaxation time ({t2_ms:.2f} ms)')
    ax_e.set_ylabel('omega error')
    ax_e.set_xlabel('Time (ms)')
    ax_e.grid(True)
    ax_e.legend()
    
    plt.suptitle(f"Frequency (omega) Estimation Errors (type={configurator.sim_type})")
    error_plot_filename = f"errors_plot_{timestamp}.png"
    error_plot_path = os.path.join(output_dir, error_plot_filename)
    plt.savefig(error_plot_path, dpi=150)
    plt.close()

    # --- Generate Markdown Report ---
    type_str = configurator.sim_type if configurator.sim_type is not None else "constant_omega"
    report_base = f"report_sim_{type_str}_dc_{configurator.dc}_tf_{configurator.tf}_{timestamp}"
    
    report_path = os.path.join(output_dir, f"{report_base}.md")
    sim_data_filename = f"simulation_data_{timestamp}.npz"
    ekf_data_filename = f"ekf_inference_data_{timestamp}.npz"
    ckf_data_filename = f"ckf_inference_data_{timestamp}.npz"
    
    abs_sim_path = os.path.abspath(os.path.join(output_dir, sim_data_filename))
    abs_ekf_path = os.path.abspath(os.path.join(output_dir, ekf_data_filename))
    abs_ckf_path = os.path.abspath(os.path.join(output_dir, ckf_data_filename))
    
    ekf_mae = np.mean(np.abs(err_ekf[:, 2]))
    ckf_mae = np.mean(np.abs(err_ckf[:, 2]))
    
    # Append CRB and PEM information if available
    crb_report_section = ""
    if crb_data is not None:
        true_f_L = configurator.w0 / (2.0 * np.pi)
        pem_freq_hz = crb_data['pem_freq_hz']
        pem_error_hz = pem_freq_hz - true_f_L
        crb_report_section = f"""
## Cramér-Rao Bounds & Prediction Error Method (Constant Omega Only)
- **Analytical Asymptotic CRB:** {crb_data['asympt_crb']:.5f} mHz
- **Monte Carlo CRB (ntries=20):** {crb_data['mc_crb']:.5f} mHz
- **PEM Frequency Estimate:** {pem_freq_hz:.5f} Hz (True: {true_f_L:.5f} Hz)
- **PEM Estimation Error:** {pem_error_hz * 1e3:.5f} mHz
"""

    report_content = f"""# Unitless Magnetometer EKF vs CKF Comparison Report

**Date/Time:** {run_time.strftime("%Y-%m-%d %H:%M:%S")}
**Simulation Type:** {configurator.sim_type}
{crb_report_section}
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

## Estimator Comparison Results
- **EKF Mean Absolute Error (MAE):** {ekf_mae:.5f}
- **CKF Mean Absolute Error (MAE):** {ckf_mae:.5f}

## Data Files
- **Simulation Data:** [simulation_data_{timestamp}.npz](file:///{abs_sim_path.replace(os.sep, '/')})
- **EKF Inference Data:** [ekf_inference_data_{timestamp}.npz](file:///{abs_ekf_path.replace(os.sep, '/')})
- **CKF Inference Data:** [ckf_inference_data_{timestamp}.npz](file:///{abs_ckf_path.replace(os.sep, '/')})

## Visualizations

### Tracking Coordinates (Simulated vs Estimated)
![Coordinates Tracking]({coord_plot_filename})

### Estimation Errors with $\pm 3\sigma$ Covariance Bounds (Omega Only)
![Estimation Errors]({error_plot_filename})
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    # --- Generate PDF Report ---
    pdf_path = os.path.join(output_dir, f"{report_base}.pdf")
    with PdfPages(pdf_path) as pdf:
        # Page 1: Text Metadata Report
        fig_text = plt.figure(figsize=(8.5, 11))
        fig_text.clf()
        
        text_content = (
            "Unitless Magnetometer EKF vs CKF Report\n\n"
            f"Date/Time: {run_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Simulation Type: {configurator.sim_type}\n\n"
            "Configuration Parameters:\n"
            f"- Number of atoms (N): {configurator.N}\n"
            f"- Parameter q: {configurator.q}\n"
            f"- Relaxation time (T2): {configurator.T2} s\n"
            f"- Coupling constant (g_D): {configurator.g_D} pA\n"
            f"- Photon noise PSD (Sph): {configurator.Sph} pA^2/Hz\n"
            f"- Mean Larmor frequency (w0): {configurator.w0} rad/s\n"
            f"- Time step (h): {configurator.h} s\n"
            f"- Probing rate factor (measure_every_nth): {configurator.measure_every_nth}\n\n"
            "Performance Comparison:\n"
            f"- EKF Mean Absolute Error (MAE): {ekf_mae:.5f}\n"
            f"- CKF Mean Absolute Error (MAE): {ckf_mae:.5f}\n\n"
        )
        if crb_data is not None:
            true_f_L = configurator.w0 / (2.0 * np.pi)
            pem_freq_hz = crb_data['pem_freq_hz']
            pem_error_hz = pem_freq_hz - true_f_L
            text_content += (
                "Cramér-Rao Bounds & PEM:\n"
                f"- Analytical Asymptotic CRB: {crb_data['asympt_crb']:.5f} mHz\n"
                f"- Monte Carlo CRB (ntries=20): {crb_data['mc_crb']:.5f} mHz\n"
                f"- PEM Freq Estimate: {pem_freq_hz:.5f} Hz\n"
                f"- PEM Estimation Error: {pem_error_hz * 1e3:.5f} mHz\n\n"
            )
        text_content += (
            "Data Files Saved in Subfolder:\n"
            f"- Simulation Data: {sim_data_filename}\n"
            f"- EKF Inference Data: {ekf_data_filename}\n"
            f"- CKF Inference Data: {ckf_data_filename}\n"
        )
        fig_text.text(0.1, 0.95, "Unitless Magnetometer EKF vs CKF Report", fontsize=16, fontweight='bold', va='top')
        fig_text.text(0.1, 0.9, text_content, fontsize=10, fontfamily='monospace', va='top')
        pdf.savefig(fig_text)
        plt.close(fig_text)
        
        # Page 2: Coordinates Plot
        fig_coord, axs_c = plt.subplots(3, 1, layout='constrained', figsize=(8.5, 11))
        axs_c[0].plot(time_arr * T2 * 1e3, xs[:, 0] / xc, label='Sim J_y', color='C0')
        axs_c[0].plot(t_meas * T2 * 1e3, x_ekf[:, 0] / xc, '--', label='EKF J_y', color='orange')
        axs_c[0].plot(t_meas * T2 * 1e3, x_ckf[:, 0] / xc, ':', label='CKF J_y', color='green')
        axs_c[0].set_ylabel('J_y')
        axs_c[0].grid(True)
        
        axs_c[1].plot(time_arr * T2 * 1e3, xs[:, 1] / xc, label='Sim J_z', color='C0')
        axs_c[1].plot(t_meas * T2 * 1e3, x_ekf[:, 1] / xc, '--', label='EKF J_z', color='orange')
        axs_c[1].plot(t_meas * T2 * 1e3, x_ckf[:, 1] / xc, ':', label='CKF J_z', color='green')
        axs_c[1].scatter(t_meas * T2 * 1e3, yh / xc, color='red', alpha=0.3, s=5, label='Meas (noisy)')
        axs_c[1].set_ylabel('J_z')
        axs_c[1].grid(True)
        
        axs_c[2].plot(time_arr * T2 * 1e3, xs[:, 2] / T2, label='Sim omega', color='C0')
        axs_c[2].plot(t_meas * T2 * 1e3, x_ekf[:, 2] / T2, '--', label='EKF omega', color='orange')
        axs_c[2].plot(t_meas * T2 * 1e3, x_ckf[:, 2] / T2, ':', label='CKF omega', color='green')
        axs_c[2].set_ylabel('omega')
        axs_c[2].set_xlabel('Time (ms)')
        axs_c[2].grid(True)
        
        for ax in axs_c:
            ax.axvline(x=t2_ms, color='red', linestyle='--', label=f'T2 relaxation time ({t2_ms:.2f} ms)')
            ax.legend()
            
        fig_coord.suptitle(f"Coordinates Tracking Comparison (type={configurator.sim_type})")
        pdf.savefig(fig_coord)
        plt.close(fig_coord)
        
        # Page 3: Errors Plot (Omega Only)
        fig_err, ax_pe = plt.subplots(layout='constrained', figsize=(8.5, 11))
        ax_pe.plot(t_meas * T2 * 1e3, err_ekf_scaled, label='EKF Error', color='orange')
        ax_pe.fill_between(
            t_meas * T2 * 1e3, 
            -3 * sigma_ekf_scaled, 
            3 * sigma_ekf_scaled, 
            color='orange', 
            alpha=0.1, 
            label='EKF +- 3-sigma bounds'
        )
        ax_pe.plot(t_meas * T2 * 1e3, err_ckf_scaled, label='CKF Error', color='green')
        ax_pe.fill_between(
            t_meas * T2 * 1e3, 
            -3 * sigma_ckf_scaled, 
            3 * sigma_ckf_scaled, 
            color='green', 
            alpha=0.1, 
            label='CKF +- 3-sigma bounds'
        )
        ax_pe.axvline(x=t2_ms, color='red', linestyle='--', label=f'T2 relaxation time ({t2_ms:.2f} ms)')
        ax_pe.set_ylabel('omega error')
        ax_pe.set_xlabel('Time (ms)')
        ax_pe.grid(True)
        ax_pe.legend()
        
        fig_err.suptitle(f"Frequency (omega) Estimation Errors Comparison (type={configurator.sim_type})")
        pdf.savefig(fig_err)
        plt.close(fig_err)
        
    logger.info(f"Report, PDF, and plots successfully generated in: {output_dir}")


def run_pipeline(
    configurator: UnitlessSimpleMagnetometerConfigurator = None,
    num_steps=20,
    data_input_path=None,
    output_dir=None
) -> str:
    """
    Main pipeline runner.
    Runs EKF and CKF inference, generates coordinates & errors plots, and outputs report.md.
    """
    if configurator is None:
        configurator = UnitlessSimpleMagnetometerConfigurator()

    run_time = datetime.now()
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = os.path.join("runs", f"run_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    sim_data_path = os.path.join(output_dir, f"simulation_data_{timestamp}.npz")
    ekf_data_path = os.path.join(output_dir, f"ekf_inference_data_{timestamp}.npz")
    ckf_data_path = os.path.join(output_dir, f"ckf_inference_data_{timestamp}.npz")

    # Step 1: Get Simulation Data (Load or Run)
    if data_input_path:
        logger.info(f"Loading simulation data from file: {data_input_path}")
        loaded = np.load(data_input_path)
        sim_data = {key: loaded[key] for key in loaded.files}
        np.savez(sim_data_path, **sim_data)
    else:
        sim_data = run_simulation(configurator, num_steps=num_steps, save_path=sim_data_path)

    # Step 2: Run CD EKF Inference
    ekf_data = run_inference(
        configurator, 
        t_meas=sim_data['t_meas'], 
        yh=sim_data['yh'], 
        save_path=ekf_data_path,
        filter_cls=CD_EKF_unitless_magnetometer,
        desc='EKF Filter'
    )

    # Step 3: Run CD CKF Inference
    ckf_data = run_inference(
        configurator, 
        t_meas=sim_data['t_meas'], 
        yh=sim_data['yh'], 
        save_path=ckf_data_path,
        filter_cls=CD_CKF_unitless_magnetometer,
        desc='CKF Filter'
    )

    # Step 4: Run CRB & PEM calculations (only for constant Larmor frequency case)
    crb_data = None
    if configurator.sim_type is None:
        logger.info("Computing Cramér-Rao Bounds and PEM estimates...")
        crb_calc = CramerRaoBounds(configurator)
        asympt_crb = crb_calc.calculate_asymptotic()
        
        # Monte Carlo CRB for the measurement size (run with 20 tries for speed in pipeline)
        nsamples = len(sim_data['yh'])
        mc_crb = crb_calc.calculate_monte_carlo(nsamples=nsamples, ntries=20)
        
        # PEM estimate
        T0 = configurator.meas_probing_rate_unitless
        sig_v = configurator.sig_v
        xc = configurator.xc
        xini = np.array([0.0, 0.5 * xc * configurator.N])
        m0_m = xini.copy()
        S0_m = np.zeros((2, 2))
        pem_estimator = PredictionErrorMethod(T0, sig_v, m0_m, S0_m)
        th_range = (configurator.w01 - 20.0, configurator.w01 + 20.0)
        th_est, _ = pem_estimator.estimate(sim_data['yh'], th_range)
        pem_freq_hz = th_est / (2.0 * np.pi * configurator.T2)
        
        crb_data = {
            'asympt_crb': asympt_crb,
            'mc_crb': mc_crb,
            'pem_freq_hz': pem_freq_hz,
            'th_est': th_est
        }

    # Step 5: Plots and Report Generation
    generate_plots_and_report(configurator, sim_data, ekf_data, ckf_data, output_dir, timestamp=timestamp, run_time=run_time, crb_data=crb_data)
    
    type_str = configurator.sim_type if configurator.sim_type is not None else "constant_omega"
    report_base = f"report_sim_{type_str}_dc_{configurator.dc}_tf_{configurator.tf}_{timestamp}"
    
    return output_dir, report_base


def run_constant_omega_case():
    """Runs a full simulation and EKF inference for constant Larmor frequency (omega)."""
    config = UnitlessSimpleMagnetometerConfigurator(
        sim_type=None,
        tf=2.0,
        dc=0.0,
        tau=0.001,
        measure_every_nth=100
    )
    result_dir, report_name = run_pipeline(
        configurator=config,
        num_steps=20,
        output_dir="runs/run_constant_omega"
    )
    print(f"Constant Omega pipeline executed successfully. View report at: {os.path.abspath(os.path.join(result_dir, f'{report_name}.md'))}")


def run_ou_process_case():
    """Runs a full simulation and EKF inference for an Ornstein-Uhlenbeck Larmor frequency process."""
    config = UnitlessSimpleMagnetometerConfigurator(
        sim_type="OU",
        tf=1.0,
        dc=0.01,
        tau=1e3,
        measure_every_nth=10
    )
    result_dir, report_name = run_pipeline(
        configurator=config,
        num_steps=20,
        output_dir="runs/run_ou_process"
    )
    print(f"OU Process pipeline executed successfully. View report at: {os.path.abspath(os.path.join(result_dir, f'{report_name}.md'))}")


def run_sine_case():
    """Runs a full simulation and EKF inference for a sinusoidal Larmor frequency process."""
    config = UnitlessSimpleMagnetometerConfigurator(
        sim_type="sine",
        tf=2.0,
        dc=0.0,
        tau=1e3,
        measure_every_nth=10
    )
    result_dir, report_name = run_pipeline(
        configurator=config,
        num_steps=20,
        output_dir="runs/run_sine"
    )
    print(f"Sine pipeline executed successfully. View report at: {os.path.abspath(os.path.join(result_dir, f'{report_name}.md'))}")


def run_jump_case():
    """Runs a full simulation and EKF inference for a piecewise constant jump (step) Larmor frequency process."""
    config = UnitlessSimpleMagnetometerConfigurator(
        sim_type="jump",
        tf=2.0,
        dc=0.0,
        tau=1e3,
        measure_every_nth=10
    )
    result_dir, report_name = run_pipeline(
        configurator=config,
        num_steps=20,
        output_dir="runs/run_jump"
    )
    print(f"Jump pipeline executed successfully. View report at: {os.path.abspath(os.path.join(result_dir, f'{report_name}.md'))}")


if __name__ == "__main__":
    # Execute all cases to compare the estimator behaviors:
    
    # 1. Constant Larmor frequency (omega) simulation and EKF tracking
    run_constant_omega_case()
    
    # 2. OU Larmor frequency process simulation and EKF tracking
    run_ou_process_case()
    
    # 3. Sinusoidal Larmor frequency simulation and EKF tracking
    run_sine_case()
    
    # 4. Piecewise constant step/jump Larmor frequency simulation and EKF tracking
    run_jump_case()
