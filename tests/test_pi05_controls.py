import unittest
import numpy as np
from g1_deploy.core import limit_gripper_controls, gripper_target, validate_chunk

class PI05ControlsTests(unittest.TestCase):
    def test_saturation_preserves_dex1_decisions_and_other_actions(self):
        a = np.zeros((50, 23)); a[:, :19] = .123
        values = [(-.2, -.01), (10.18, -.00645), (10.04, 1.01), (4.99, .49), (5., .5)]
        for i, (trigger, squeeze) in enumerate(values):
            a[i, 19:] = [trigger, squeeze, trigger, squeeze]
        b = limit_gripper_controls(a, 50)
        np.testing.assert_array_equal(a[:, :19], b[:, :19])
        self.assertGreater(a[1, 19], 10)  # input was not mutated
        for i in range(50):
            self.assertEqual(gripper_target(2., 2.1, *a[i, 19:21]), gripper_target(2., 2.1, *b[i, 19:21]))
        self.assertTrue((b[:, 19:] >= 0).all())
        self.assertTrue((b[:, 19:] <= [10, 1, 10, 1]).all())

    def test_bad_outputs_and_arm_outliers_remain_rejected(self):
        for bad in [np.zeros((49, 23)), np.full((50, 23), np.nan), np.full((50, 23), np.inf)]:
            with self.assertRaises(ValueError):
                limit_gripper_controls(bad, 50)
        a = np.zeros((50, 23)); a[0, 7] = 100
        with self.assertRaisesRegex(ValueError, 'left_x'):
            validate_chunk(limit_gripper_controls(a, 50), [-1]*23, [1]*23, 50)
