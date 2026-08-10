import unittest
import numpy as np
from types import SimpleNamespace

from space_state_model.simple_sensor_model import Simple_Sensor_Model
from space_state_model.Bell_Bloom_model import Bell_Bloom_Magnetometer_Model
from space_state_model.corr_simple_sensor_model import Simple_CC_Correlated_Sensor_Model
from space_state_model.corr_10x10_model import Corr_10x10_CC_Sensor_Model


class TestStateSpaceModels(unittest.TestCase):
    def setUp(self):
        # Create standard parameters
        self.num_atoms = 1000.0
        self.params = SimpleNamespace(
            x_0=np.array([1.0, 2.0, 3.0]),
            dt=0.01,
            T2=0.87,
            num_atoms=self.num_atoms,
            omega_pumping=2.0 * np.pi,
            pump_amplitude=1.0,
            noise=SimpleNamespace(
                Q_jx=0.01,
                Q_jy=0.01,
                Q_freq=0.0,
                Q_m=0.01
            ),
            measurement=SimpleNamespace(
                measurement_strength=1.0,
                H=np.array([[0.0, 1.0, 0.0]]),
                noise=SimpleNamespace(
                    R=0.01,
                    mean=0.0
                )
            )
        )

    def test_simple_sensor_model(self):
        model = Simple_Sensor_Model(t=0.0, simulation_params=self.params)
        self.assertEqual(model.t, 0.0)
        np.testing.assert_array_equal(model.x, self.params.x_0)

        # Test step
        x_next, z_next = model.step(method="default")
        self.assertEqual(len(x_next), 3)
        self.assertEqual(len(z_next), 1)
        self.assertAlmostEqual(model.t, 0.01)

    def test_bell_bloom_magnetometer_model(self):
        model = Bell_Bloom_Magnetometer_Model(t=0.0, simulation_params=self.params)
        self.assertEqual(model.t, 0.0)
        np.testing.assert_array_equal(model.x, self.params.x_0)

        # Test step
        x_next, z_next = model.step(method="default")
        self.assertEqual(len(x_next), 3)
        self.assertEqual(len(z_next), 1)
        self.assertAlmostEqual(model.t, 0.01)

    def test_simple_cc_correlated_sensor_model(self):
        model = Simple_CC_Correlated_Sensor_Model(t=0.0, simulation_params=self.params)
        self.assertEqual(model.t, 0.0)
        np.testing.assert_array_equal(model.x, self.params.x_0)

        # Test step with naive/default method
        x_next, z_next = model.step(method="default")
        self.assertEqual(len(x_next), 3)
        self.assertEqual(len(z_next), 1)
        self.assertAlmostEqual(model.t, 0.01)

        # Test step with odeint method
        model2 = Simple_CC_Correlated_Sensor_Model(t=0.0, simulation_params=self.params)
        x_next_ode, z_next_ode = model2.step(method="odeint")
        self.assertEqual(len(x_next_ode), 3)
        self.assertEqual(len(z_next_ode), 1)

    def test_corr_10x10_cc_sensor_model(self):
        params_10x10 = SimpleNamespace(
            x_0=np.zeros(10),
            dt=0.01,
            measurement_strength=1.0,
            coll_decoherence=0.1,
            eta=0.5,
            num_atoms=self.num_atoms,
            H=np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        )
        model = Corr_10x10_CC_Sensor_Model(t=0.0, simulation_params=params_10x10)
        self.assertEqual(model.t, 0.0)
        np.testing.assert_array_equal(model.x, params_10x10.x_0)

        # Test step
        x_next, z_next = model.step(method="default")
        self.assertEqual(len(x_next), 10)
        self.assertEqual(len(z_next), 1)
        self.assertAlmostEqual(model.t, 0.01)


if __name__ == '__main__':
    unittest.main()
