import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1_deploy.catalog import ROOT, robot_config


class PortableConfigTests(unittest.TestCase):
    def test_examples_enable_watchdogs_and_preserve_model_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            config = robot_config('galaxea-fold-towel', ROOT / 'config/robot.example.json')
        for key in ('command_heartbeat_enabled', 'action_plan_expiry_enabled',
                    'robot_state_expiry_enabled'):
            self.assertTrue(config[key])
        self.assertEqual(config['request_timeout_s'], 10)
        self.assertEqual(config['action_bounds_mode'], 'clip')
        self.assertEqual(Path(config['wbc_repo']), ROOT / 'vendor/wbc_teleoperation')

    def test_local_file_and_explicit_path_precedence(self):
        example = json.loads((ROOT / 'config/robot.example.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config').mkdir()
            (root / 'config/robot.example.json').write_text(json.dumps(example))
            local = dict(example, interface='local0')
            (root / 'config/robot.json').write_text(json.dumps(local))
            explicit = root / 'explicit.json'
            explicit.write_text(json.dumps(dict(example, interface='explicit0')))
            with patch('g1_deploy.catalog.ROOT', root), patch.dict(os.environ, {}, clear=True), \
                    patch('g1_deploy.catalog.model_spec', return_value={
                        'model_path': 'unused', 'task': 'test', 'chunk_size': 32,
                        'family': 'groot', 'task_slug': 'test', 'head_mode': 'stereo_full'}):
                self.assertEqual(robot_config('test')['interface'], 'local0')
                with patch.dict(os.environ, {'ROBOT_CONFIG': str(explicit)}):
                    self.assertEqual(robot_config('test')['interface'], 'explicit0')
                    with patch.dict(os.environ, {'G1_INTERFACE': 'override0', 'WBC_REPO': '/tmp/wbc'}):
                        config = robot_config('test')
                        self.assertEqual(config['interface'], 'override0')
                        self.assertEqual(config['wbc_repo'], '/tmp/wbc')
                (root / 'config/robot.json').unlink()
                self.assertEqual(robot_config('test')['interface'], example['interface'])
