import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from g1_deploy.rollout_store import inventory, delete_rollout
from g1_deploy.web_ui import Console, handler_for


class RolloutStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'outputs/rollouts'
        self.run = self.root / 'by-model/groot/carry-box/test-execute'
        self.run.mkdir(parents=True)
        (self.run / 'manifest.json').write_text(json.dumps({
            'format': 'g01-rollout-v1', 'start_unix_s': 100,
            'mode': 'execute', 'config': {'task': 'carrying box', 'model_key': 'groot-carry-box'}}))
        (self.run / 'events.jsonl').write_text('{}\n')
        self.id = self.run.relative_to(self.root).as_posix()

    def test_inventory_incomplete_and_delete_alias(self):
        alias = self.root / 'by-name/alias'
        alias.parent.mkdir()
        alias.symlink_to(self.run, target_is_directory=True)
        (self.root / 'index.json').write_text('{}')
        rows = inventory(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['task'], 'carry-box')
        self.assertFalse(rows[0]['summary_present'])
        delete_rollout(self.root, self.id, self.run.name)
        self.assertFalse(self.run.exists())
        self.assertFalse(alias.is_symlink())
        self.assertFalse((self.root / 'index.json').exists())
        self.assertEqual(inventory(self.root), [])

    def test_reject_traversal_parent_and_bad_confirmation(self):
        for path, name in [('../elsewhere', 'elsewhere'), (str(self.run), self.run.name),
                           ('by-model/groot/carry-box', 'carry-box'), (self.id, 'wrong'),
                           ('./' + self.id, self.run.name), ('', ''), (None, None)]:
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                delete_rollout(self.root, path, name)
        self.assertTrue(self.run.exists())

    def test_reject_symlink_and_do_not_follow_payload_links(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.run, target_is_directory=True)
        with self.assertRaises(ValueError):
            delete_rollout(self.root, 'alias', 'alias')
        outside = Path(self.tmp.name) / 'keep.txt'
        outside.write_text('keep')
        (self.run / 'link').symlink_to(outside)
        delete_rollout(self.root, self.id, self.run.name)
        self.assertEqual(outside.read_text(), 'keep')

    def test_active_and_external_clients_block_deletion(self):
        console = Console()
        with patch('g1_deploy.web_ui.ROOT', Path(self.tmp.name)), \
             patch.object(console.client, 'group_alive', return_value=True), \
             patch('g1_deploy.web_ui.recording_processes', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'Stop the robot client'):
                console.remove_rollout({'id': self.id, 'confirm_name': self.run.name})
        with patch('g1_deploy.web_ui.recording_processes', return_value=[123]):
            with self.assertRaisesRegex(ValueError, 'Stop the robot client'):
                console.remove_rollout({'id': self.id, 'confirm_name': self.run.name})
        self.assertTrue(self.run.exists())

    def test_http_list_confirmation_and_delete(self):
        console = Console()
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(console))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}/api/rollouts'
        try:
            with patch('g1_deploy.web_ui.ROOT', Path(self.tmp.name)), \
                 patch('g1_deploy.web_ui.recording_processes', return_value=[]):
                self.assertEqual(json.load(urllib.request.urlopen(url))['rollouts'][0]['id'], self.id)
                request = urllib.request.Request(url + '/delete',
                    data=json.dumps({'id': self.id}).encode(), headers={'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as err:
                    urllib.request.urlopen(request)
                self.assertEqual(err.exception.code, 400)
                self.assertTrue(self.run.exists())
                request.data = json.dumps({'id': self.id, 'confirm_name': self.run.name}).encode()
                request.add_header('Origin', 'https://untrusted.example')
                with self.assertRaises(urllib.error.HTTPError) as err:
                    urllib.request.urlopen(request)
                self.assertEqual(err.exception.code, 403)
                request.remove_header('Origin')
                result = json.load(urllib.request.urlopen(request))
                self.assertEqual(result['deleted'], self.id)
                self.assertFalse(self.run.exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
