import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock
from http.server import ThreadingHTTPServer
import urllib.request
import urllib.error

from g1_deploy.web_ui import Runner, Console, handler_for, validate_request


class UiTests(unittest.TestCase):
    @patch('g1_deploy.web_ui.robot_config', return_value={'interface': 'eth0'})
    def test_validation(self, config):
        data = dict(model='test')
        self.assertEqual(validate_request(data, {'test': {}}, ['eth0']), ('test', 'execute', 'eth0', 0))
        for change in [dict(model='unknown'), dict(mode='policy')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(dict(data, **change), {'test': {}}, ['eth0'])
        self.assertEqual(validate_request(dict(data, mode='execute'), {'test': {}}, ['eth0']), ('test', 'execute', 'eth0', 0))
        # Ignore stale controls left in a previously loaded page.
        self.assertEqual(validate_request(dict(data, duration=30, interface='other'), {'test': {}}, ['eth0']), ('test', 'execute', 'eth0', 0))
        with self.assertRaises(ValueError):
            validate_request(data, {'test': {}}, ['other'])

    def test_api_without_token(self):
        runner = Console()
        runner.schema = lambda: None
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(runner))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = 'http://127.0.0.1:%s' % server.server_port
        try:
            self.assertEqual(json.load(urllib.request.urlopen(url + '/api/status'))['client']['state'], 'idle')
            request = urllib.request.Request(url + '/api/client/stop', data=b'{}', headers={'Content-Type': 'application/json'})
            self.assertEqual(json.load(urllib.request.urlopen(request))['client']['state'], 'idle')
            request.add_header('Origin', 'https://other.example')
            with self.assertRaises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(request)
            self.assertEqual(err.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_server_client_lifecycle(self):
        console = Console()
        console.client = Mock()
        console.policy = Mock()
        console.client.group_alive.return_value = False
        console.client.stopping = False
        console.policy.stopping = False
        console.policy.status.return_value = {'active': True, 'state': 'running'}
        console.client.status.return_value = {'active': False, 'state': 'idle'}
        console.schema = Mock(return_value=None)
        with self.assertRaises(ValueError):
            console.execute({'model': 'test'})
        console.client.start.assert_not_called()
        console.schema.return_value = {'model_key': 'other'}
        with self.assertRaises(ValueError):
            console.execute({'model': 'test'})
        console.schema.return_value = {'model_key': 'test'}
        console.execute({'model': 'test'})
        console.client.start.assert_called_once_with({'model': 'test'})
        console.stop_client()
        console.policy.stop.assert_not_called()
        console.client.group_alive.return_value = True
        with self.assertRaises(ValueError):
            console.stop_server()
        console.client.group_alive.return_value = False
        console.stop_server()
        console.policy.stop.assert_called_once()
        # Loading server does not need a live robot interface.
        self.assertEqual(validate_request({'model': 'test'}, {'test': {}}, [], 'policy'), ('test', 'policy', '', 0))

    def test_launch_duplicate_and_stop_entire_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            child_marker = root / 'child-stopped'
            # A real parent and grandchild, with no robot/model dependencies.
            child_code = f'import signal,time,pathlib; signal.signal(signal.SIGINT,lambda *_: (pathlib.Path({str(child_marker)!r}).touch(),exit(0))); print("ready",flush=True); time.sleep(60)'
            parent_code = 'import signal,subprocess,sys,time; signal.signal(signal.SIGINT,signal.default_int_handler); p=subprocess.Popen([sys.executable,"-c",sys.argv[1]]); p.wait()'
            import shlex
            (root / 'scripts/run_model.sh').write_text('exec ' + shlex.join([sys.executable, '-c', parent_code, child_code]) + '\n')
            (root / 'unused').mkdir()
            (root / 'unused/model.safetensors').write_bytes(b'weights')
            runner = Runner('policy')
            with patch('g1_deploy.web_ui.ROOT', root), patch('g1_deploy.web_ui.catalog', return_value={'models': {'test': {'model_path': 'unused', 'weight_bytes': 7}}}), patch('g1_deploy.web_ui.robot_config', return_value={'interface': socket.if_nameindex()[0][1]}):
                data = dict(model='test')
                try:
                    self.assertTrue(runner.start(data)['active'])
                    with self.assertRaises(ValueError):
                        runner.start(data)
                    deadline = time.monotonic() + 5
                    while 'ready' not in runner.status()['log'] and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertIn('ready', runner.status()['log'])
                    runner.stop()
                    runner.close()
                    self.assertTrue(child_marker.exists())
                    self.assertFalse(runner.status()['active'])
                finally:
                    runner.close()


if __name__ == '__main__':
    unittest.main()
