import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from g1_deploy.catalog import model_downloaded
from g1_deploy.lingbot_backend import validate_robot_mapping, restore_constant_actions


def mapping():
    def entry(key, source, slices):
        return {key: {'origin_keys': [{source: dict(start=a, end=b)} for a, b in slices],
                      'subtract_state': False}}
    return {
        'states': [entry('observation.state.arm.position', 'observation.state', [(15, 22), (22, 29)]),
                   entry('observation.state.waist.position', 'observation.state', [(12, 15)]),
                   entry('observation.state.hand.position', 'observation.state', [(0, 12)])],
        'actions': [entry('action.base.position', 'action', [(0, 3)]),
                    entry('action.waist.position', 'action', [(3, 7)]),
                    entry('action.end.position', 'action', [(7, 13), (13, 19)]),
                    entry('action.hand.position', 'action', [(19, 23)])],
        'images': [{f'observation.images.{a}': {'origin_keys': f'observation.images.{b}'}}
                   for a, b in [('camera_top', 'head'), ('camera_wrist_left', 'left_wrist'),
                                ('camera_wrist_right', 'right_wrist')]],
    }


class VlaTests(unittest.TestCase):
    def test_unavailable_galaxea_rejected_before_backend_loading(self):
        from g1_deploy.multi_policy import make_backend
        with patch('g1_deploy.multi_policy.model_spec', return_value={
                'family': 'galaxea', 'unavailable_reason': 'G05 asset validation failed'}), \
             patch('g1_deploy.multi_policy.UniversalBackend') as backend:
            with self.assertRaisesRegex(ValueError, 'asset validation failed'):
                make_backend('galaxea-fold-towel', 'cpu')
            backend.assert_not_called()

    def test_restore_only_exact_training_constants_and_preserve_raw(self):
        raw = np.array([[0.740101, 2.0], [0.7399, -2.0]])
        original = raw.copy()
        restored, fixed = restore_constant_actions(raw, [0.74, -1.0], [0.74, 1.0])
        self.assertEqual(fixed, [0])
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(restored[:, 0], [0.74, 0.74])
        np.testing.assert_array_equal(restored[:, 1], raw[:, 1])
        with self.assertRaises(ValueError):
            restore_constant_actions([[np.nan, 0]], [0.74, -1], [0.74, 1])

    def test_invalid_checkpoint_cannot_start_from_console(self):
        from g1_deploy.web_ui import Runner
        models = {'bad': {'unavailable_reason': 'Invalid source weights'}}
        with patch('g1_deploy.web_ui.catalog', return_value={'models': models}), \
             patch('g1_deploy.web_ui.socket.if_nameindex', return_value=[]), \
             patch('g1_deploy.web_ui.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'Invalid source weights'):
                Runner(mode='policy').start({'model': 'bad'})
            launch.assert_not_called()

    def test_sharded_readiness_requires_all_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {'model_path': '.', 'weight_bytes': 8, 'required_files': [
                {'path': 'shard1', 'size': 4}, {'path': 'shard2', 'size': 4},
                {'path': 'norm_stats.json', 'size': 2}]}
            (root / 'shard1').write_bytes(b'1234')
            self.assertFalse(model_downloaded(spec, root))
            (root / 'shard2').write_bytes(b'1234')
            self.assertFalse(model_downloaded(spec, root))
            (root / 'norm_stats.json').write_bytes(b'{}')
            self.assertTrue(model_downloaded(spec, root))
            (root / 'shard2').write_bytes(b'12')
            self.assertFalse(model_downloaded(spec, root))

    def test_mapping_rejects_relative_reordered_or_wrong_sources(self):
        original = mapping()
        validate_robot_mapping(original)
        bad = copy.deepcopy(original)
        bad['actions'][0]['action.base.position']['subtract_state'] = True
        with self.assertRaises(ValueError):
            validate_robot_mapping(bad)
        bad = copy.deepcopy(original)
        bad['states'][0]['observation.state.arm.position']['origin_keys'].reverse()
        with self.assertRaises(ValueError):
            validate_robot_mapping(bad)
        bad = copy.deepcopy(original)
        bad['images'][0]['observation.images.camera_top']['origin_keys'] = 'observation.images.left_wrist'
        with self.assertRaises(ValueError):
            validate_robot_mapping(bad)


if __name__ == '__main__':
    unittest.main()
