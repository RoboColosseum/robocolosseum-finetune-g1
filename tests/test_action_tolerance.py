import unittest
import numpy as np
from g1_deploy.core import validate_chunk

class ToleranceTests(unittest.TestCase):
    def test_clips_both_sides_without_mutating_input(self):
        a = np.zeros((50,23)); a[0,0]=1.19; a[1,0]=-1.19
        b=validate_chunk(a,[-1]*23,[1]*23,50,overshoot_fraction=.1)
        self.assertEqual(b[0,0],1); self.assertEqual(b[1,0],-1)
        self.assertEqual(a[0,0],1.19)
        validate_chunk(b,[-1]*23,[1]*23,50)
        a[0,0]=1.201
        with self.assertRaises(ValueError): validate_chunk(a,[-1]*23,[1]*23,50,overshoot_fraction=.1)

    def test_actual_height_failure_and_strict_client(self):
        lo=np.full(23,-1.); hi=np.ones(23); lo[6]=.405; hi[6]=.74
        a=np.zeros((50,23)); a[:,6]=.741939
        with self.assertRaises(ValueError): validate_chunk(a,lo,hi,50)
        b=validate_chunk(a,lo,hi,50,overshoot_fraction=.1)
        self.assertTrue((b[:,6]==.74).all())
        validate_chunk(b,lo,hi,50)

    def test_invalid_and_constant_bounds(self):
        for a in (np.zeros((49,23)),np.full((50,23),np.nan),np.full((50,23),np.inf)):
            with self.assertRaises(ValueError): validate_chunk(a,[-1]*23,[1]*23,50,overshoot_fraction=.1)
        a=np.zeros((50,23)); a[0,0]=.001
        with self.assertRaises(ValueError): validate_chunk(a,[0]*23,[0]*23,50,overshoot_fraction=.1)

    def test_server_returns_raw_outliers_but_rejects_invalid_data(self):
        from g1_deploy.policy_server import GrootBackend
        backend = object.__new__(GrootBackend)
        backend.length = 50
        a = np.full((50, 23), 1000.)
        np.testing.assert_array_equal(backend.validate_actions(a), a)
        for invalid in (np.zeros((49, 23)), np.full((50, 23), np.nan), np.full((50, 23), np.inf)):
            with self.assertRaises(ValueError): backend.validate_actions(invalid)

    def test_clip_mode_handles_recorded_spikes_without_mutating_input(self):
        a = np.zeros((32, 23)); a[22, 1] = 1.2007039785; a[4, 1] = -.6166447401
        lo = np.full(23, -.48); hi = np.full(23, .48)
        b = validate_chunk(a, lo, hi, bounds_mode='clip')
        self.assertEqual(b[22, 1], .48)
        self.assertEqual(b[4, 1], -.48)
        self.assertEqual(a[22, 1], 1.2007039785)
        np.testing.assert_array_equal(b[:, 0], a[:, 0])
        with self.assertRaises(ValueError):
            validate_chunk(a, lo, hi, overshoot_fraction=.1)

    def test_clip_mode_still_rejects_invalid_outputs_and_bounds(self):
        for a in (np.zeros((31,23)), np.full((32,23),np.nan), np.full((32,23),np.inf)):
            with self.assertRaises(ValueError):
                validate_chunk(a, [-1]*23, [1]*23, bounds_mode='clip')
        for lo, hi in (([1]*23, [-1]*23), ([float('nan')]*23, [1]*23)):
            with self.assertRaises(ValueError):
                validate_chunk(np.zeros((32,23)), lo, hi, bounds_mode='clip')

    def test_only_g05_profiles_enable_clip(self):
        from g1_deploy.catalog import catalog, robot_config
        for key, spec in catalog()['models'].items():
            self.assertEqual(robot_config(key)['action_bounds_mode'],
                             'clip' if spec['family']=='galaxea' else 'reject')
