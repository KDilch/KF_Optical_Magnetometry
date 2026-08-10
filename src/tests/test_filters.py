import unittest
import numpy as np
from types import SimpleNamespace

from kalman_filter.simple_model_ekf import MagnetometerEKF
from kalman_filter.simple_model_cd_ekf import CD_EKF_simple_magnetometer
from kalman_filter.bell_bloom_cd_ekf import CD_Bell_Bloom_EKF
from kalman_filter.bell_bloom_ekf import BellBloomEKF
from kalman_filter.simple_model_ukf import SimpleMagnetometerUKF
from kalman_filter.simple_model_integrated import MagnetometerEKFIntegrated
from kalman_filter.corr_simple_model_ekf import CorrSimpleModelEKF


class TestKalmanFilters(unittest.TestCase):
    def setUp(self):
        self.num_atoms = 1000.0
        self.omega_pump = 2.0 * np.pi * 29.07
        self.x_0 = np.array([0.0, self.num_atoms / 2.0, self.omega_pump])
        
        self.filter_params = SimpleNamespace(
            dt=0.0005,
            T2=0.87,
            omega_pumping=self.omega_pump,
            pump_amplitude=100.0,
            num_atoms=self.num_atoms,
            inference_method='RK23',
            x_0=self.x_0.copy(),
            t_0=0.0,
            P0=np.array([[10.0, 0.0, 0.0],
                         [0.0, 10.0, 0.0],
                         [0.0, 0.0, 100.0]]),
            noise=SimpleNamespace(
                Q=np.array([[2.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 0.0]])
            ),
            measurement=SimpleNamespace(
                measurement_strength=1e-9,
                H=np.array([[0.0, 1.0, 0.0]]),
                dim_z=1,
                R=np.array([[0.01]])
            )
        )

    def test_magnetometer_ekf(self):
        ekf = MagnetometerEKF(model_params=self.filter_params)
        self.assertEqual(ekf.model_params.t_0, 0.0)
        np.testing.assert_array_equal(ekf.x_est, self.x_0)

        # Test single predict_update step
        z = 0.5 * 1e-9 * (self.num_atoms / 2.0)
        ekf.predict_update(z)
        self.assertEqual(len(ekf.x_est), 3)
        self.assertEqual(ekf.P_est.shape, (3, 3))

    def test_cd_ekf_simple_magnetometer(self):
        ekf = CD_EKF_simple_magnetometer(model_params=self.filter_params)
        self.assertEqual(ekf.model_params.t_0, 0.0)
        np.testing.assert_array_equal(ekf.x_est, self.x_0)

        # Test predict_update step
        z = np.array([0.5 * 1e-9 * (self.num_atoms / 2.0)])
        ekf.predict_update(z)
        self.assertEqual(len(ekf.x_est), 3)
        self.assertEqual(ekf.P_est.shape, (3, 3))

    def test_cd_bell_bloom_ekf(self):
        ekf = CD_Bell_Bloom_EKF(model_params=self.filter_params)
        self.assertEqual(ekf.model_params.t_0, 0.0)
        np.testing.assert_array_equal(ekf.x_est, self.x_0)

        # Test predict_update step
        z = np.array([0.5 * 1e-9 * (self.num_atoms / 2.0)])
        ekf.predict_update(z)
        self.assertEqual(len(ekf.x_est), 3)
        self.assertEqual(ekf.P_est.shape, (3, 3))

    def test_bell_bloom_ekf(self):
        ekf = BellBloomEKF(model_params=self.filter_params)
        self.assertEqual(ekf.model_params.t_0, 0.0)
        np.testing.assert_array_equal(ekf.x_est, self.x_0)

        # Test predict_update step
        z = 0.5 * 1e-9 * (self.num_atoms / 2.0)
        ekf.predict_update(z)
        self.assertEqual(len(ekf.x_est), 3)
        self.assertEqual(ekf.P_est.shape, (3, 3))

    def test_simple_magnetometer_ukf(self):
        ukf = SimpleMagnetometerUKF(model_params=self.filter_params)
        np.testing.assert_array_equal(ukf.x_est, self.x_0)

        # Test predict_update step
        z = np.array([0.5 * 1e-9 * (self.num_atoms / 2.0)])
        ukf.predict_update(z)
        self.assertEqual(len(ukf.x_est), 3)
        self.assertEqual(ukf.P_est.shape, (3, 3))

    def test_magnetometer_ekf_integrated(self):
        # The integrated EKF uses state vector [J_x, J_y, omega, damping] of size 4
        # We need a 4D config
        x_0_4d = np.array([0.0, self.num_atoms / 2.0, self.omega_pump, 1.0 / 0.87])
        filter_params_4d = SimpleNamespace(
            dt=0.0005,
            T2=0.87,
            N=self.num_atoms,
            omega_pumping=self.omega_pump,
            pump_amplitude=100.0,
            inference_method='RK23',
            x_0=x_0_4d,
            t_0=0.0,
            P0=np.eye(4) * 10.0,
            noise=SimpleNamespace(
                Q=np.eye(4) * 2.0
            ),
            measurement=SimpleNamespace(
                measurement_strength=1e-9,
                H=np.array([[0.0, 1.0, 0.0, 0.0]]),
                dim_z=1,
                R=np.array([[0.01]])
            )
        )
        ekf = MagnetometerEKFIntegrated(model_params=filter_params_4d)
        np.testing.assert_array_equal(ekf.x_est, x_0_4d)

        # Test predict_update step
        z = 0.5 * 1e-9 * (self.num_atoms / 2.0)
        ekf.predict_update(z)
        self.assertEqual(len(ekf.x_est), 4)
        self.assertEqual(ekf.P_est.shape, (4, 4))

    def test_corr_simple_model_ekf(self):
        # Correlated EKF needs noise.B and noise.S
        filter_params_corr = SimpleNamespace(
            dt=0.0005,
            T2=0.87,
            omega_pumping=self.omega_pump,
            pump_amplitude=100.0,
            inference_method='RK23',
            x_0=self.x_0.copy(),
            t_0=0.0,
            P0=np.array([[10.0, 0.0, 0.0],
                         [0.0, 10.0, 0.0],
                         [0.0, 0.0, 100.0]]),
            noise=SimpleNamespace(
                Q=np.array([[0.01]]),
                B=np.array([[1.0], [1.0], [0.0]]),
                S=np.array([[0.01]])
            ),
            measurement=SimpleNamespace(
                measurement_strength=1e-9,
                H=np.array([[0.0, 1.0, 0.0]]),
                dim_z=1,
                R=np.array([[0.01]])
            )
        )
        ekf = CorrSimpleModelEKF(model_params=filter_params_corr)
        np.testing.assert_array_equal(ekf.x_est, self.x_0)

        # Test predict_update step
        z = 0.5 * 1e-9 * (self.num_atoms / 2.0)
        ekf.predict_update(z)
        self.assertEqual(len(ekf.x_est), 3)
        self.assertEqual(ekf.P_est.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
