"""Solver accuracy and smoothing lag must remain distinct."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from g1_deploy.wbc_client import Kinematics, ArmTrackingError, IKConvergenceError, load_config

ROOT = Path(__file__).resolve().parents[1]

class IKResidualTests(unittest.TestCase):
    def make_kinematics(self, raw, filtered):
        k = Kinematics.__new__(Kinematics)
        model = types.SimpleNamespace(lowerPositionLimit=np.full(14, -2.), upperPositionLimit=np.full(14, 2.))
        k.ik = types.SimpleNamespace(
            solve_ik=lambda *args: (filtered.copy(), np.zeros(14)),
            opti=types.SimpleNamespace(stats=lambda: {'success': True}, value=lambda _: raw.copy()),
            var_q=object(),
            solve_fk=lambda q: (np.array([q[0], 0, 0, 0, 0, 0]), np.zeros(6)),
            reduced_robot=types.SimpleNamespace(model=model, data=None),
        )
        return k

    def test_filtered_lag_is_not_solver_failure_and_slew_is_retained(self):
        config = load_config(ROOT / 'config/deploy.example.json')
        k = self.make_kinematics(np.zeros(14), np.full(14, .2))
        pin = types.SimpleNamespace(rnea=lambda *args: np.zeros(14))
        with patch.dict('sys.modules', {'pinocchio': pin}):
            command, _ = k.command(np.zeros(23), np.zeros(14), np.zeros(14), np.zeros(14),
                                   np.asarray(config['initial_base_cmd']), config, 1/30)
        np.testing.assert_allclose(command['arm_q'], config['arm_speed_rad_s']/30)

    def test_bad_raw_solution_still_rejected_and_diagnosed(self):
        config = load_config(ROOT / 'config/deploy.example.json')
        with tempfile.TemporaryDirectory() as root:
            config['ik_failure_dir'] = root
            k = self.make_kinematics(np.full(14, .101), np.zeros(14))
            with self.assertRaisesRegex(RuntimeError, 'left raw IK residual too large'):
                k.command(np.zeros(23), np.zeros(14), np.zeros(14), np.zeros(14),
                          np.asarray(config['initial_base_cmd']), config, 1/30)
            report = json.loads(next(Path(root).glob('*.json')).read_text())
            self.assertEqual(report['residual_stage'], 'raw_ik_solution')
            self.assertAlmostEqual(report['raw_errors']['left']['position_m'], .101)
            self.assertEqual(report['filtered_errors']['left']['position_m'], 0)

    def test_nonconvergence_is_recoverable(self):
        config = load_config(ROOT / 'config/deploy.example.json')
        k = self.make_kinematics(np.zeros(14), np.zeros(14))
        k.ik.opti.stats = lambda: {'success': False}
        with self.assertRaises(IKConvergenceError):
            k.command(np.zeros(23), np.zeros(14), np.zeros(14), np.zeros(14),
                      np.asarray(config['initial_base_cmd']), config, 1/30)

    def test_filtered_command_tracking_guard_remains(self):
        config = load_config(ROOT / 'config/deploy.example.json')
        k = self.make_kinematics(np.zeros(14), np.ones(14))
        with self.assertRaises(ArmTrackingError):
            k.command(np.zeros(23), np.zeros(14), np.zeros(14), np.ones(14),
                      np.asarray(config['initial_base_cmd']), config, 1/30)
