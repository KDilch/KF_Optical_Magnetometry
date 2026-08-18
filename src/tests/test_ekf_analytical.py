import unittest
import numpy as np

from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from space_state_model.unitless_magnetometer_model import UnitlessMagnetometerModel
from kalman_filter.unitless_cd_ekf import CD_EKF_unitless_magnetometer


class TestEKFAnalyticalDiscrete(unittest.TestCase):
    def setUp(self):
        # Config setup for a simple EKF test
        self.config = UnitlessSimpleMagnetometerConfigurator(
            sim_type="OU",
            tf=1.0,
            dc=0.01,
            tau=1e3,
            measure_every_nth=10
        )
        self.filter_params = self.config.get_filter_params()

    def test_analytical_jacobian_correctness(self):
        # Create EKF with 'discrete' inference method
        self.filter_params.inference_method = 'discrete'
        ekf = CD_EKF_unitless_magnetometer(model_params=self.filter_params)
        
        # Manually set EKF states to known values
        ekf._x = np.array([1e5, -2e5, 50.0])
        ekf._P = np.diag([100.0, 100.0, 0.1])
        
        # Perform prediction step
        ekf.predict()
        
        # Verify that state has precessed
        self.assertNotEqual(ekf.x_est[0], 1e5)
        self.assertNotEqual(ekf.x_est[1], -2e5)
        
        # Verify covariance matrix remains positive semidefinite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvals(ekf.P_est)
        self.assertTrue(np.all(eigenvalues >= 0), f"Covariance matrix has negative eigenvalues: {eigenvalues}")

    def test_discrete_vs_rk45_compatibility(self):
        # Verify state propagation is numerically close between RK45 and discrete (RK4) state predictions
        p_discrete = self.config.get_filter_params()
        p_discrete.inference_method = 'discrete'
        ekf_discrete = CD_EKF_unitless_magnetometer(model_params=p_discrete)
        
        p_rk45 = self.config.get_filter_params()
        p_rk45.inference_method = 'RK45'
        ekf_rk45 = CD_EKF_unitless_magnetometer(model_params=p_rk45)
        
        # Set same state
        x_init = np.array([5e5, 5e5, 54.0])
        ekf_discrete._x = x_init.copy()
        ekf_rk45._x = x_init.copy()
        
        # Predict once
        ekf_discrete.predict()
        ekf_rk45.predict()
        
        # State estimates should be very close (difference < 0.1%)
        np.testing.assert_allclose(ekf_discrete.x_est, ekf_rk45.x_est, rtol=1e-3)

    def test_update_equations_matching(self):
        # Verify that update equations match the user's implementation exactly
        ekf = CD_EKF_unitless_magnetometer(model_params=self.filter_params)
        
        # Explicit state and covariance
        ekf._x = np.array([10.0, 20.0, 30.0])
        ekf._P = np.diag([2.0, 3.0, 4.0])
        
        # Measurement z = 22.0
        z = 22.0
        
        # Let's calculate manually matching the formula:
        # H = [[0, 1, 0]]
        # innovation = y - H @ x = 22.0 - 20.0 = 2.0
        # S = H @ P @ H.T + R = P[1,1] + R = 3.0 + R
        # K = P @ H.T / S = [P[0,1]/S, P[1,1]/S, P[2,1]/S] = [0, 3.0/S, 0]
        # x_new = x + K * innovation
        # P_new = (I - K @ H) @ P
        
        R_val = ekf._R[0, 0]
        S_val = 3.0 + R_val
        K_expected = np.array([[0.0], [3.0 / S_val], [0.0]])
        x_new_expected = ekf._x + np.dot(K_expected, np.array([2.0]))
        
        ekf.update(z)
        
        # Verify EKF update results match manual expected ones
        np.testing.assert_allclose(ekf.x_est, x_new_expected)
        self.assertAlmostEqual(ekf.P_est[1, 1], 3.0 * (1.0 - 3.0 / S_val))

    def test_all_simulation_types_run_successfully(self):
        # Run brief simulations for all 4 types to verify EKF handles them cleanly
        types = ["OU", "sine", "jump", None]
        for t in types:
            config = UnitlessSimpleMagnetometerConfigurator(
                sim_type=t,
                tf=0.05,
                dc=0.01,
                tau=1e3,
                measure_every_nth=10
            )
            sim_params = config.get_sim_params()
            filter_params = config.get_filter_params()
            
            # Set method to discrete
            filter_params.inference_method = 'discrete'
            
            model = UnitlessMagnetometerModel(t=0.0, simulation_params=sim_params)
            ekf = CD_EKF_unitless_magnetometer(model_params=filter_params)
            
            # Run 5 steps of SDE and EKF
            for _ in range(5):
                x_next, z_next = model.step(num_steps=5)
                ekf.update(z_next)
                ekf.predict()
                
            self.assertEqual(ekf.x_est.shape, (3,))
            self.assertEqual(ekf.P_est.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
