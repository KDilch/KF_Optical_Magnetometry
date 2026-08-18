# -*- coding: utf-8 -*-
import unittest
import numpy as np
from configs.unitless_magnetometer_params import UnitlessSimpleMagnetometerConfigurator
from evaluation.crb_pem import (
    get_AG,
    simulate_discrete_system,
    PredictionErrorMethod,
    CramerRaoBounds
)

class TestCRBPEM(unittest.TestCase):
    def setUp(self):
        # A test configurator with standard physical params
        self.config = UnitlessSimpleMagnetometerConfigurator(
            N=0.44 * 1e12,
            q=0.198,
            T2=0.87 * 1e-3,
            g_D=0.00177,
            Sph=96.0,
            w0=2 * np.pi * 1e4,
            h=50 * 1e-9,
            tf=0.1,  # Short run for speed
            dc=0.0,
            tau=0.001,
            measure_every_nth=10,  # Finer measurement to reduce likelihood landscape complexity
            sim_type=None
        )
        self.th0 = self.config.w01
        self.T0 = self.config.meas_probing_rate_unitless
        self.sig_v = self.config.sig_v
        self.xc = self.config.xc
        self.x_init = np.array([0.0, 0.5 * self.xc * self.config.N])

    def test_get_AG_matrices(self):
        A, G, D = get_AG(self.th0, self.T0)
        # Verify shapes
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(G.shape, (2, 2))
        self.assertEqual(D.shape, (2, 2))
        
        # Verify symmetry of D
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        
        # Verify A is close to rotation matrix scaled by exp(-T0)
        c = np.cos(self.th0 * self.T0)
        s = np.sin(self.th0 * self.T0)
        expected_A = np.exp(-self.T0) * np.array([[c, s], [-s, c]])
        np.testing.assert_allclose(A, expected_A, atol=1e-12)

    def test_simulate_discrete_system(self):
        nsamples = 100
        y, x_hist = simulate_discrete_system(nsamples, self.th0, self.T0, self.sig_v, self.x_init)
        
        self.assertEqual(len(y), nsamples)
        self.assertEqual(x_hist.shape, (nsamples, 2))
        
        # Initial condition matching
        np.testing.assert_allclose(x_hist[0, :], self.x_init, atol=1e-12)

    def test_get_lfun_behavior(self):
        nsamples = 50
        y, _ = simulate_discrete_system(nsamples, self.th0, self.T0, self.sig_v, self.x_init)
        
        # Evaluated at true theta vs away from true theta
        m_init = self.x_init.copy()
        S_init = np.zeros((2, 2))
        
        pem = PredictionErrorMethod(self.T0, self.sig_v, m_init, S_init)
        val_true = pem.get_likelihood(self.th0, y)
        val_away = pem.get_likelihood(self.th0 + 5.0, y)
        
        # Verify likelihood functions return valid floats
        self.assertTrue(isinstance(val_true, float))
        self.assertTrue(isinstance(val_away, float))
        
        # Likelihood should be finite
        self.assertTrue(np.isfinite(val_true))
        self.assertTrue(np.isfinite(val_away))

    def test_estimate_theta_pem(self):
        # We need a clean simulation with low measurement noise to guarantee PEM converges close to th0
        clean_config = UnitlessSimpleMagnetometerConfigurator(
            N=0.44 * 1e12,
            q=0.198,
            T2=0.87 * 1e-3,
            g_D=0.00177,
            Sph=0.96,  # 100x lower noise PSD
            w0=2 * np.pi * 1e4,
            h=50 * 1e-9,
            tf=0.5,
            dc=0.0,
            tau=0.001,
            measure_every_nth=10,
            sim_type=None
        )
        
        th0 = clean_config.w01
        T0 = clean_config.meas_probing_rate_unitless
        sig_v = clean_config.sig_v
        xc = clean_config.xc
        x_init = np.array([0.0, 0.5 * xc * clean_config.N])
        
        # Simulate clean-ish trajectory
        np.random.seed(42)  # For reproducibility in tests
        y, _ = simulate_discrete_system(100, th0, T0, sig_v, x_init)
        
        m_init = x_init.copy()
        S_init = np.zeros((2, 2))
        
        pem = PredictionErrorMethod(T0, sig_v, m_init, S_init)
        th_range = (th0 - 5.0, th0 + 5.0)
        th_est, fval = pem.estimate(y, th_range)
        
        # PEM should estimate close to true th0
        self.assertAlmostEqual(th_est, th0, delta=0.5)

    def test_asymptotic_crb(self):
        crb = CramerRaoBounds(self.config)
        asympt_crb = crb.calculate_asymptotic()
        self.assertGreater(asympt_crb, 0.0)
        
        # If N increases, asympt CRB should decrease (more information)
        config_more_atoms = UnitlessSimpleMagnetometerConfigurator(
            N=self.config.N * 2,
            q=self.config.q,
            T2=self.config.T2,
            g_D=self.config.g_D,
            Sph=self.config.Sph,
            w0=self.config.w0,
            h=self.config.h,
            tf=self.config.tf,
            dc=self.config.dc,
            tau=self.config.tau,
            measure_every_nth=self.config.measure_every_nth,
            sim_type=None
        )
        crb_more = CramerRaoBounds(config_more_atoms)
        asympt_crb_more = crb_more.calculate_asymptotic()
        self.assertLess(asympt_crb_more, asympt_crb)

    def test_monte_carlo_crb(self):
        crb = CramerRaoBounds(self.config)
        crb_mc = crb.calculate_monte_carlo(nsamples=10, ntries=2)
        self.assertGreater(crb_mc, 0.0)
        self.assertTrue(np.isfinite(crb_mc))

if __name__ == '__main__':
    unittest.main()
