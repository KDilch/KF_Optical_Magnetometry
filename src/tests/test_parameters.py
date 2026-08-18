import unittest
import numpy as np

from configs.unitless_magnetometer_params import (
    UnitlessSimpleMagnetometerConfigurator,
    UnitlessSimulationParams,
    UnitlessFilterParams
)
from space_state_model.unitless_magnetometer_model import UnitlessMagnetometerModel
from kalman_filter.unitless_cd_ekf import CD_EKF_unitless_magnetometer


class TestUnitlessMagnetometerParameters(unittest.TestCase):
    def setUp(self):
        self.configurator = UnitlessSimpleMagnetometerConfigurator(
            N=0.44 * 1e12,
            q=0.198,
            T2=0.87 * 1e-3,
            g_D=0.00177,
            Sph=96.0,
            w0=2 * np.pi * 1e4,
            h=50 * 1e-9,
            tf=2.0,
            dc=0.01,
            tau=0.001,
            measure_every_nth=100,
            sim_type="OU"
        )

    def test_simulation_params_derivation(self):
        sim_params = self.configurator.get_sim_params()
        self.assertIsInstance(sim_params, UnitlessSimulationParams)
        
        # Test values are derived correctly
        self.assertAlmostEqual(sim_params.dt, (50 * 1e-9) / (0.87 * 1e-3))
        self.assertAlmostEqual(sim_params.T2, 0.87 * 1e-3)
        self.assertAlmostEqual(sim_params.w01, 2 * np.pi * 1e4 * 0.87 * 1e-3)
        self.assertEqual(sim_params.type, "OU")
        self.assertEqual(len(sim_params.x_0), 3)

    def test_filter_params_derivation(self):
        filter_params = self.configurator.get_filter_params()
        self.assertIsInstance(filter_params, UnitlessFilterParams)
        
        # Test nested structure compatibility for CD_EKF
        self.assertTrue(hasattr(filter_params, 'noise'))
        self.assertTrue(hasattr(filter_params, 'measurement'))
        
        self.assertIsInstance(filter_params.noise.Q, np.ndarray)
        self.assertEqual(filter_params.noise.Q.shape, (3, 3))
        self.assertIsInstance(filter_params.measurement.R, np.ndarray)
        self.assertEqual(filter_params.measurement.dim_z, 1)

    def test_model_and_filter_instantiation(self):
        sim_params = self.configurator.get_sim_params()
        filter_params = self.configurator.get_filter_params()
        
        # Test model can be instantiated and stepped
        model = UnitlessMagnetometerModel(t=0.0, simulation_params=sim_params)
        x_next, z_next = model.step(num_steps=5)
        self.assertEqual(x_next.shape, (3,))
        
        # Test filter can be instantiated and stepped
        ekf = CD_EKF_unitless_magnetometer(model_params=filter_params)
        np.testing.assert_array_equal(ekf.x_est, filter_params.x_0)
        
        # Test EKF predict/update
        ekf.predict()
        ekf.update(z_next)
        self.assertEqual(ekf.x_est.shape, (3,))


if __name__ == '__main__':
    unittest.main()
