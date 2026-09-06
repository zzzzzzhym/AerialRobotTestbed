import unittest
import numpy as np

from drone import parameters
from inflow_model.blade_params import APC_8x6
from learning.bemt_traditional_fit.bemt_model import BemtModel


class TestBemtModelMath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fitter = BemtModel(APC_8x6(), parameters.PennStateARILab550())

    def test_compute_body_drag_force_zero_k(self):
        self.fitter.k_body_drag = 0.0
        force = self.fitter.compute_body_drag_force(
            v_i_avg=3.0,
            u_free_avg=np.array([5.0, 0.0, 0.0]),
            r_disk=np.eye(3),
        )
        np.testing.assert_array_equal(force, np.zeros(3))

    def test_compute_body_drag_force_horizontal_disk_no_wind(self):
        # r_disk=I → disk_z = world z; no wind → v_total = v_i_avg; force = k * v_i^2 * z_hat
        self.fitter.k_body_drag = 1.0
        force = self.fitter.compute_body_drag_force(
            v_i_avg=2.0,
            u_free_avg=np.zeros(3),
            r_disk=np.eye(3),
        )
        np.testing.assert_array_almost_equal(force, np.array([0.0, 0.0, 4.0]))

    def test_compute_body_drag_force_downward_wind_increases_drag(self):
        # Wind pointing downward (-z) augments downwash; drag magnitude should be larger
        self.fitter.k_body_drag = 1.0
        r_disk = np.eye(3)
        v_i = 2.0
        force_no_wind = self.fitter.compute_body_drag_force(v_i, np.zeros(3), r_disk)
        force_with_wind = self.fitter.compute_body_drag_force(v_i, np.array([0.0, 0.0, -1.0]), r_disk)
        self.assertGreater(force_with_wind[2], force_no_wind[2])

    def test_compute_residual_force_hover_equilibrium(self):
        # f_inertial exactly balances gravity → residual = 0
        self.fitter.k_body_drag = 0.0
        m = self.fitter.params.m
        g = 9.81
        f_inertial = np.array([0.0, 0.0, m * g])
        residual = self.fitter.compute_residual_force(f_inertial, a_groundtruth=np.zeros(3))
        np.testing.assert_array_almost_equal(residual, np.zeros(3))

    def test_compute_residual_force_upward_acceleration(self):
        # f_inertial = m*(g + a_z) in z → residual = 0
        self.fitter.k_body_drag = 0.0
        m = self.fitter.params.m
        g = 9.81
        a_z = 2.0
        f_inertial = np.array([0.0, 0.0, m * (g + a_z)])
        residual = self.fitter.compute_residual_force(
            f_inertial, a_groundtruth=np.array([0.0, 0.0, a_z])
        )
        np.testing.assert_array_almost_equal(residual, np.zeros(3))

    def test_compute_residual_force_nonzero_means_underthrust(self):
        # Provide less thrust than needed → residual is negative in z
        self.fitter.k_body_drag = 0.0
        m = self.fitter.params.m
        g = 9.81
        f_inertial = np.array([0.0, 0.0, m * g * 0.9])  # 10% short
        residual = self.fitter.compute_residual_force(f_inertial, a_groundtruth=np.zeros(3))
        self.assertLess(residual[2], 0.0)


class TestConfigureForBodyDragFit(unittest.TestCase):

    def _make_model(self):
        from learning.bemt_traditional_fit.fitting_config import ModelConfig
        model_config = ModelConfig(
            coarse_n_elements=2, coarse_n_rotation_segments=6, coarse_sample_distance=100,
            fine_n_elements=20, fine_n_rotation_segments=18, fine_sample_distance=20,
        )
        return BemtModel(APC_8x6(), parameters.PennStateARILab550(), model_config)

    def test_bounds_becomes_one_dimensional(self):
        model = self._make_model()
        model.configure_for_body_drag_fit()
        self.assertEqual(model.BOUNDS, [(0.0, 10.0)])

    def test_parameter_names_becomes_k_body_drag_only(self):
        model = self._make_model()
        model.configure_for_body_drag_fit()
        self.assertEqual(model.PARAMETER_NAMES, ("k_body_drag",))

    def test_class_level_bounds_unchanged(self):
        original_class_bounds = BemtModel.BOUNDS
        model = self._make_model()
        model.configure_for_body_drag_fit()
        self.assertIs(BemtModel.BOUNDS, original_class_bounds)

    def test_class_level_parameter_names_unchanged(self):
        original_names = BemtModel.PARAMETER_NAMES
        model = self._make_model()
        model.configure_for_body_drag_fit()
        self.assertIs(BemtModel.PARAMETER_NAMES, original_names)

    def test_other_model_instance_unaffected(self):
        model_a = self._make_model()
        model_b = self._make_model()
        model_a.configure_for_body_drag_fit()
        self.assertEqual(model_b.BOUNDS, BemtModel.BOUNDS)


if __name__ == "__main__":
    unittest.main()
