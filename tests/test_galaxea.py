import unittest
from unittest.mock import patch
from g1_deploy.web_ui import Console, validate_request


class G05WebTests(unittest.TestCase):
    def test_g05_timeout_reaches_generated_client_config(self):
        from g1_deploy.catalog import catalog, robot_config
        models = catalog()['models']
        for key, spec in models.items():
            if spec['family'] == 'galaxea':
                self.assertEqual(robot_config(key)['request_timeout_s'], 10.0)
        self.assertEqual(robot_config('groot-carry-box')['request_timeout_s'], 2.0)

    def test_invalid_model_timeout_rejected(self):
        from g1_deploy.catalog import model_spec, robot_config
        for value in [0, -1, float('nan'), float('inf')]:
            spec = dict(model_spec('galaxea-carry-box'), request_timeout_s=value)
            with patch('g1_deploy.catalog.model_spec', return_value=spec):
                with self.assertRaises(ValueError):
                    robot_config('galaxea-carry-box')

    def test_replay_does_not_require_robot_interface(self):
        self.assertEqual(validate_request({'model': 'g05'}, {'g05': {}}, [], 'test'),
                         ('g05', 'test', '', 0))
        with self.assertRaises(ValueError):
            validate_request({'model': 'g05', 'mode': 'execute'}, {'g05': {}}, [], 'test')

    def test_replay_rejects_active_robot_and_wrong_model(self):
        c = Console()
        with patch.object(c.client, 'group_alive', return_value=True), patch.object(c.test, 'start') as start:
            with self.assertRaises(ValueError): c.test_policy({'model': 'g05'})
            start.assert_not_called()
        with patch.object(c, 'schema', return_value={'model_key': 'other'}), patch.object(c.test, 'start') as start:
            with self.assertRaises(ValueError): c.test_policy({'model': 'g05'})
            start.assert_not_called()

    def test_replay_prevents_execute_and_server_stop(self):
        c = Console()
        with patch.object(c.test, 'group_alive', return_value=True), patch.object(c.client, 'start') as start:
            with self.assertRaises(ValueError): c.execute({'model': 'g05'})
            with self.assertRaises(ValueError): c.stop_server()
            start.assert_not_called()

    def test_replay_uses_test_runner(self):
        c = Console()
        with patch.object(c, 'schema', return_value={'model_key': 'g05'}), patch.object(c.test, 'start') as test, patch.object(c.client, 'start') as client:
            c.test_policy({'model': 'g05'})
            test.assert_called_once_with({'model': 'g05'})
            client.assert_not_called()
