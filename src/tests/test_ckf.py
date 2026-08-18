import unittest
import numpy as np

from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from space_state_model.unitless_magnetometer_model import UnitlessMagnetometerModel
from kalman_filter.unitless_cd_ckf import CD_CKF_unitless_magnetometer


class TestCDCKFUnitlessMagnetometer(unittest.TestCase):
    def setUp(self):
        self.config = UnitlessSimpleMagnetometerConfigurator(
            sim_type="OU",
            tf=1.0,
            dc=0.01,
            tau=1e3,
            measure_every_nth=10
        )
        self.filter_params = self.config.get_filter_params()

    def test_ckf_initialization(self):
        ckf = CD_CKF_unitless_magnetometer(model_params=self.filter_params)
        self.assertEqual(ckf.n, 3)
        self.assertEqual(len(ckf.wghts), 6)
        np.testing.assert_allclose(ckf.wghts, np.ones(6) / 6.0)
        self.assertEqual(ckf.pts.shape, (3, 6))

    def test_ckf_prediction_step(self):
        ckf = CD_CKF_unitless_magnetometer(model_params=self.filter_params)
        
        # Explicit states and covariance
        ckf._x = np.array([1e5, -1e5, 54.0])
        ckf._P = np.diag([100.0, 100.0, 0.1])
        
        # Predict once
        ckf.predict()
        
        # Spin values must change due to precession
        self.assertNotEqual(ckf.x_est[0], 1e5)
        self.assertNotEqual(ckf.x_est[1], -1e5)
        
        # Covariance eigenvalues must remain non-negative
        eigenvalues = np.linalg.eigvals(ckf.P_est)
        self.assertTrue(np.all(eigenvalues >= 0.0), f"Negative eigenvalues found: {eigenvalues}")

    def test_ckf_update_step(self):
        ckf = CD_CKF_unitless_magnetometer(model_params=self.filter_params)
        
        # Explicit states and covariance
        ckf._x = np.array([10.0, 20.0, 30.0])
        ckf._P = np.diag([2.0, 3.0, 4.0])
        
        # Measurement z = 22.0
        z = 22.0
        
        R_val = ckf._R[0, 0]
        S_val = 3.0 + R_val
        K_expected = np.array([[0.0], [3.0 / S_val], [0.0]])
        x_new_expected = ckf._x + np.dot(K_expected, np.array([2.0]))
        
        ckf.update(z)
        
        # Verify update calculations match expectation
        np.testing.assert_allclose(ckf.x_est, x_new_expected)
        self.assertAlmostEqual(ckf.P_est[1, 1], 3.0 * (1.0 - 3.0 / S_val))

    def test_ckf_loop_run(self):
        sim_params = self.config.get_sim_params()
        model = UnitlessMagnetometerModel(t=0.0, simulation_params=sim_params)
        ckf = CD_CKF_unitless_magnetometer(model_params=self.filter_params)
        
        for _ in range(10):
            x_next, z_next = model.step(num_steps=5)
            ckf.update(z_next)
            ckf.predict()
            
        self.assertEqual(ckf.x_est.shape, (3,))
        self.assertEqual(ckf.P_est.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
