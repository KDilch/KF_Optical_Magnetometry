# KF_Optical_Magnetometry

A high-performance Python framework for real-time continuous-discrete inference of optical magnetometer parameters. This package implements Continuous-Discrete Extended Kalman Filters (CD-EKF) and Continuous-Discrete Cubature Kalman Filters (CD-CKF), alongside Cramér-Rao Bound (CRB) statistical limits and Prediction Error Method (PEM) parameter estimators, to track atomic spin polarization and estimate the Larmor frequency.

---

## Features

- **Continuous-Discrete Filters**:
  - **CD-EKF**: Implements analytical linearization for continuous state propagation (SDE) and discrete updates.
  - **CD-CKF**: Uses third-degree spherical-radial cubature points to propagate states through nonlinear spin dynamics without requiring Jacobian derivations.
- **Spin Polarization Models**:
  - state-space models describing unitless angular momentum ($J_x, J_y, J_z$) and Larmor frequency dynamics ($\omega$).
  - Supports multiple Larmor frequency process types: constant frequency, Ornstein-Uhlenbeck (OU) diffusion, sinusoidal drift, and piecewise constant step-jumps.
- **Statistical Bounds & Parameter Estimation**:
  - **Cramér-Rao Bound (CRB)**: Evaluates the numerical Fisher Information matrix via finite-difference log-likelihoods over Monte Carlo trials.
  - **Asymptotic CRB**: Analytical steady-state limit of estimation variance as $t \to \infty$.
  - **Prediction Error Method (PEM)**: Estimates constant Larmor frequency via bounded optimization of the Kalman Filter innovation log-likelihood.
- **Validation Suite**:
  - Automated unit tests covering parameter derivations, filter correctness, model propagation, and analytical bounds.
  - Scripts comparing EKF and CKF estimation errors and standard deviation bounds.

---

## Directory Structure

```
kalman_filters_in_magnetometry/
├── README.md                           # Main project documentation
├── example_data/                       # Pre-generated example runs & bounds comparison
│   ├── sim_ou.npz                      # Ornstein-Uhlenbeck simulation trajectory & filters
│   ├── sim_sine.npz                    # Sinusoidal drift simulation & filters
│   ├── sim_jump.npz                    # Step-jump simulation & filters
│   └── bounds_comparison.npz           # MC CRB, Asymptotic CRB, and PEM RMSE values
├── src/
│   ├── configs/
│   │   └── unitless_magnetometer_params.py  # Configuration class & unitless transformations
│   ├── space_state_model/
│   │   ├── model.py                    # Base abstract class for state-space models
│   │   └── unitless_magnetometer_model.py   # Magnetometer SDE state propagation
│   ├── kalman_filter/
│   │   ├── cd_ekf.py / cd_ckf.py       # Base Continuous-Discrete EKF and CKF filters
│   │   └── unitless_cd_ekf.py / unitless_cd_ckf.py # Specialized unitless filter implementations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── crb_pem.py                  # Likelihood, PEM optimization, and CRB bound calculations
│   ├── tests/
│   │   ├── test_parameters.py          # Validates configurations & scaling parameters
│   │   ├── test_ekf_analytical.py      # Analytical Jacobian & propagation checks
│   │   ├── test_ckf.py                 # CD-CKF initialization & updates
│   │   └── test_crb_pem.py             # Validates PEM, get_AG, get_lfun, and CRBs
│   ├── run_cd_ekf_unitless.py           # Runs pipelines & compares EKF/CKF performances
│   ├── generate_example_data.py        # Generates baseline datasets stored in example_data/
│   ├── benchmark_ckf_n.py              # Performance vs sample size benchmarks
│   └── run_tests.py                    # Automated test discovery runner
```

---

## Getting Started

### 1. Running the Pipeline & Generating Reports

Run the main comparison script to simulate the physical spin states, execute CD-EKF and CD-CKF tracking filters, and generate comparison reports containing estimation errors plotted against $\pm 3\sigma$ filter-calculated bounds:

```bash
python src/run_cd_ekf_unitless.py
```

Outputs will be saved in `runs/run_<timestamp>/` and include:
- `simulation_data_<timestamp>.npz`
- `ekf_inference_data_<timestamp>.npz`
- `ckf_inference_data_<timestamp>.npz`
- `report_sim_<case>_dc_<dc>_tf_<tf>_<timestamp>.md`

### 2. Generating the Example Datasets

To re-generate the baseline datasets included in `example_data/`:

```bash
python src/generate_example_data.py
```

### 3. Running Unit Tests

Run the automated test runner to verify package health:

```bash
python src/run_tests.py
```

---

## Structure of Pre-Generated Example Data

All pre-generated files are stored in [`example_data/`](file:///c:/Users/Klaudia/Documents/Python_projects/kalman_filters_in_magnetometry/example_data).

1. **`sim_ou.npz`**, **`sim_sine.npz`**, and **`sim_jump.npz`**:
   Contains dictionary variables with arrays for:
   - `time_arr`: Time grid array of simulation steps.
   - `xs`: True state vector trajectories ($J_y, J_z, \omega$).
   - `t_meas`: Timestamps of discrete measurement events.
   - `yh`: Noisy unitless measurements ($J_z + \text{noise}$).
   - `x_sim_meas`: True state values at measurement timestamps.
   - `x_ekf` / `x_ckf`: Estimated states from CD-EKF and CD-CKF.
   - `P_ekf` / `P_ckf`: Covariance diagonal elements from CD-EKF and CD-CKF.

2. **`bounds_comparison.npz`**:
   Contains evaluation results of statistical Larmor frequency estimation limits:
   - `nsamples_range`: Tested range of measurement steps (10 to 100).
   - `crb_vals`: Monte Carlo CRB values (in mHz).
   - `asympt_crb`: Steady-state analytical CRB values (in mHz).
   - `pem_rmse_vals`: Root-Mean-Square error of PEM frequency estimates over multiple trials (in mHz).
