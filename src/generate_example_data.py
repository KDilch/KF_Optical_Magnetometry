# -*- coding: utf-8 -*-
import os
import numpy as np
import logging
from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from space_state_model.unitless_magnetometer_model import UnitlessMagnetometerModel
from kalman_filter.unitless_cd_ekf import CD_EKF_unitless_magnetometer
from kalman_filter.unitless_cd_ckf import CD_CKF_unitless_magnetometer
from evaluation.crb_pem import (
    simulate_discrete_system,
    PredictionErrorMethod,
    CramerRaoBounds
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_single_simulation_and_filters(configurator):
    """
    Runs a single simulation run and estimates the state using both CD-EKF and CD-CKF.
    """
    # 1. Run simulation
    sim_params = configurator.get_sim_params()
    h1 = configurator.h1
    t_max = configurator.tf / (configurator.T2 * 1e3)
    measure_every_nth = configurator.measure_every_nth
    
    model = UnitlessMagnetometerModel(t=0.0, simulation_params=sim_params)
    time_arr = np.arange(0, t_max, h1)
    xs = np.zeros((len(time_arr), 3))
    
    yh = []
    t_meas = []
    x_sim_meas = []
    
    for index, time in enumerate(time_arr):
        x, _ = model.step(num_steps=20)
        xs[index] = x
        if index % measure_every_nth == 0:
            yh.append(model.read_sensor(with_noise=True))
            t_meas.append(time)
            x_sim_meas.append(x)
            
    yh = np.array(yh)
    t_meas = np.array(t_meas)
    x_sim_meas = np.array(x_sim_meas)
    
    # 2. Run EKF Inference
    ekf_params = configurator.get_filter_params()
    ekf = CD_EKF_unitless_magnetometer(model_params=ekf_params)
    x_ekf = np.zeros((len(yh), 3))
    P_ekf = np.zeros((len(yh), 3))
    
    for index in range(len(t_meas)):
        ekf.update(yh[index])
        x_ekf[index] = ekf.x_est
        P_ekf[index] = [ekf.P_est[0, 0], ekf.P_est[1, 1], ekf.P_est[2, 2]]
        ekf.predict()
        
    # 3. Run CKF Inference
    ckf_params = configurator.get_filter_params()
    ckf_params.inference_method = 'discrete'
    ckf = CD_CKF_unitless_magnetometer(model_params=ckf_params)
    x_ckf = np.zeros((len(yh), 3))
    P_ckf = np.zeros((len(yh), 3))
    
    for index in range(len(t_meas)):
        ckf.update(yh[index])
        x_ckf[index] = ckf.x_est
        P_ckf[index] = [ckf.P_est[0, 0], ckf.P_est[1, 1], ckf.P_est[2, 2]]
        ckf.predict()
        
    return {
        'time_arr': time_arr,
        'xs': xs,
        't_meas': t_meas,
        'yh': yh,
        'x_sim_meas': x_sim_meas,
        'x_ekf': x_ekf,
        'P_ekf': P_ekf,
        'x_ckf': x_ckf,
        'P_ckf': P_ckf
    }

def main():
    # Make directory for example data
    data_dir = "example_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Set seed for reproducible data generation
    np.random.seed(1234)
    
    # --- 1. Generate OU simulation data ---
    logger.info("Generating OU process simulation example data...")
    config_ou = UnitlessSimpleMagnetometerConfigurator(
        sim_type="OU",
        tf=1.0,
        dc=0.01,
        tau=0.001,
        measure_every_nth=10
    )
    ou_data = run_single_simulation_and_filters(config_ou)
    np.savez(os.path.join(data_dir, "sim_ou.npz"), **ou_data)
    
    # --- 2. Generate Sine simulation data ---
    logger.info("Generating sinusoidal process simulation example data...")
    config_sine = UnitlessSimpleMagnetometerConfigurator(
        sim_type="sine",
        tf=1.0,
        dc=0.0,
        tau=0.001,
        measure_every_nth=10
    )
    sine_data = run_single_simulation_and_filters(config_sine)
    np.savez(os.path.join(data_dir, "sim_sine.npz"), **sine_data)
    
    # --- 3. Generate Jump simulation data ---
    logger.info("Generating step-jump process simulation example data...")
    config_jump = UnitlessSimpleMagnetometerConfigurator(
        sim_type="jump",
        tf=1.0,
        dc=0.0,
        tau=0.001,
        measure_every_nth=10
    )
    jump_data = run_single_simulation_and_filters(config_jump)
    np.savez(os.path.join(data_dir, "sim_jump.npz"), **jump_data)
    
    # --- 4. Generate Bounds Comparison data ---
    logger.info("Generating bounds comparison data (CRB, Asympt CRB, PEM)...")
    
    # Setup configurator for constant Larmor frequency comparison
    config_const = UnitlessSimpleMagnetometerConfigurator(
        sim_type=None,
        tf=1.0,
        dc=0.0,
        tau=0.001,
        measure_every_nth=100
    )
    
    # We will evaluate bounds over a range of sample sizes (number of measurement steps)
    nsamples_range = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ntries = 50
    
    crb_vals = []
    pem_mse_vals = []
    
    # Derived params
    T0 = config_const.meas_probing_rate_unitless
    sig_v = config_const.sig_v
    xc = config_const.xc
    th0 = config_const.w01
    xini = np.array([0.0, 0.5 * xc * config_const.N])
    m0_m = xini.copy()
    S0_m = np.zeros((2, 2))
    
    er_coeff = 1e3 / (2.0 * np.pi * config_const.T2)
    th_range = (th0 - 20.0, th0 + 20.0)
    
    # Instantiate Cramer-Rao bounds and PEM estimator
    crb_calculator = CramerRaoBounds(config_const)
    pem_estimator = PredictionErrorMethod(T0, sig_v, m0_m, S0_m)
    
    # Calculate Asymptotic CRB (constant value for fixed physical configuration)
    asympt_crb = crb_calculator.calculate_asymptotic()
    
    logger.info(f"Computing MC CRB and PEM MSE over {len(nsamples_range)} sample size points...")
    for ns in nsamples_range:
        logger.info(f"Running bounds for nsamples = {ns}")
        # MC CRB
        crb_val = crb_calculator.calculate_monte_carlo(nsamples=ns, ntries=ntries)
        crb_vals.append(crb_val)
        
        # PEM MSE
        pem_errors = []
        for _ in range(ntries):
            y, _ = simulate_discrete_system(ns, th0, T0, sig_v, xini)
            th_est, _ = pem_estimator.estimate(y, th_range)
            freq_error = er_coeff * (th_est - th0)
            pem_errors.append(freq_error**2)
            
        pem_mse_vals.append(np.sqrt(np.mean(pem_errors))) # RMSE in mHz
        
    bounds_data = {
        'nsamples_range': nsamples_range,
        'crb_vals': np.array(crb_vals),
        'asympt_crb': np.full_like(nsamples_range, asympt_crb, dtype=float),
        'pem_rmse_vals': np.array(pem_mse_vals)
    }
    
    np.savez(os.path.join(data_dir, "bounds_comparison.npz"), **bounds_data)
    logger.info("Example data successfully generated and saved to: " + data_dir)

if __name__ == '__main__':
    main()
