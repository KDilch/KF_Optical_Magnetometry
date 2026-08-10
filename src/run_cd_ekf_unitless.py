# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from space_state_model.unitless_magnetometer_model import UnitlessMagnetometerModel
from kalman_filter.unitless_cd_ekf import CD_EKF_unitless_magnetometer


def run_unitless_magnetometer_simulation_and_ekf(
    sim_type=None,
    tf=2.0,
    dc=0.01,
    tau=0.001,
    num_steps=20,
    measure_every_nth=100,
    save_path=None,
    N=0.44 * 1e12,
    q=0.198,
    T2=0.87 * 1e-3,
    g_D=0.00177,
    Sph=96.0,
    w0=2 * np.pi * 1e4,
    h=50 * 1e-9
):
    """
    OO execution of unitless simulation and continuous-discrete EKF.
    Supports sim_type: None, "OU", "jump", "sine"
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    configurator = UnitlessSimpleMagnetometerConfigurator(
        N=N, q=q, T2=T2, g_D=g_D, Sph=Sph, w0=w0, h=h, tf=tf, dc=dc, tau=tau,
        measure_every_nth=measure_every_nth, sim_type=sim_type
    )
    sim_params = configurator.get_sim_params()
    h1 = configurator.h1
    t_max = tf / (configurator.T2 * 1e3)
    xc = configurator.xc
    T2 = configurator.T2
    
    # 2. Initialize the OO Simulation Model
    model = UnitlessMagnetometerModel(t=0.0, simulation_params=sim_params, logger=logger)
    
    # 3. Create time grids
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
        
    # 4. Generate measurements at every nth probing step
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
    
    # 5. Initialize OO EKF Filter
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
        
    logger.info("Plotting results...")
    fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(8, 8))
    
    # Jy plot
    axs[0].plot(time_arr * T2 * 1e3, xs[:, 0] / xc, label='Sim J_y')
    axs[0].plot(t_meas * T2 * 1e3, x_est[:, 0] / xc, '--', label='EKF J_y')
    axs[0].set_ylabel('J_y')
    axs[0].grid(True)
    axs[0].legend()
    
    # Jz plot
    axs[1].plot(time_arr * T2 * 1e3, xs[:, 1] / xc, label='Sim J_z')
    axs[1].plot(t_meas * T2 * 1e3, x_est[:, 1] / xc, '--', label='EKF J_z')
    axs[1].scatter(t_meas * T2 * 1e3, yh / xc, color='red', alpha=0.3, s=5, label='Meas (noisy)')
    axs[1].set_ylabel('J_z')
    axs[1].grid(True)
    axs[1].legend()
    
    # Frequency plot
    axs[2].plot(time_arr * T2 * 1e3, xs[:, 2] / T2, label='Sim omega')
    axs[2].plot(t_meas * T2 * 1e3, x_est[:, 2] / T2, '--', label='EKF omega')
    axs[2].set_ylabel('omega')
    axs[2].set_xlabel('Time (ms)')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.suptitle(f"Unitless Magnetometer EKF Tracking (type={sim_type})")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # Example: Run EKF for an Ornstein-Uhlenbeck process signal
    run_unitless_magnetometer_simulation_and_ekf(sim_type="OU", tf=1.0, dc=1e-2, tau=1e3, num_steps=20, measure_every_nth=100)
