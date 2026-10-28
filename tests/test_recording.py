import json
import os
from pathlib import Path
import tempfile
import signal
import time
import unittest
from unittest.mock import patch

from g1_deploy.recording import RolloutRecorder


class RecordingTests(unittest.TestCase):
    def test_payload_response_and_stop_are_flushed(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {'task': 'plate into rack'}, {'action_shape': [40, 23]}, 'dry-run')
            payload = {'state': [0] * 29, 'images': {'head': 'jpeg-base64'}, 'task': 'plate into rack'}
            recorder.record('policy_request', request_id=1, payload=payload, state_time=123)
            recorder.record('policy_response', request_id=1, result={'actions': [[0] * 23] * 40})
            recorder.record('dry_run_command', request_id=1, action_index=8, command={'arm_q': [0] * 14})
            recorder.record('stop', reason='RuntimeError', detail='IK residual too large')
            recorder.close()
            events = [json.loads(line) for line in (recorder.path / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events], ['policy_request', 'policy_response', 'dry_run_command', 'stop'])
            self.assertEqual(json.loads((recorder.path / events[0]['payload_file']).read_text()), payload)
            self.assertEqual(events[2]['request_id'], events[1]['request_id'])
            summary = json.loads((recorder.path / 'summary.json').read_text())
            self.assertTrue(summary['flush_complete'])
            self.assertEqual(summary['writer_backend'], 'process')
            self.assertNotEqual(summary['writer_pid'], os.getpid())
            self.assertEqual(summary['accepted'], summary['written'])
            self.assertEqual(summary['dropped'], {})
            self.assertEqual(json.loads((recorder.path / 'manifest.json').read_text())['mode'], 'dry-run')

    def wait_writer(self, recorder):
        deadline = time.monotonic() + 5
        while not (recorder.path / 'events.jsonl').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue((recorder.path / 'events.jsonl').exists())

    def test_full_queue_drops_explicitly_without_blocking(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {}, None, 'observe', queue_size=1)
            try:
                self.wait_writer(recorder)
                os.kill(recorder.process.pid, signal.SIGSTOP)
                started = time.monotonic()
                # A stopped writer cannot service the queue; enqueue must never wait.
                with patch('g1_deploy.recording.json.dumps', side_effect=AssertionError('JSON on control thread')):
                    for i in range(1000):
                        recorder.record('observation', request_id=i, payload={})
                self.assertLess(time.monotonic() - started, .5)
                self.assertGreaterEqual(recorder.dropped['observation'], 998)
            finally:
                os.kill(recorder.process.pid, signal.SIGCONT)
                recorder.close()
            summary = json.loads((recorder.path / 'summary.json').read_text())
            self.assertTrue(summary['flush_complete'])
            self.assertEqual(sum(summary['written'].values()) + sum(summary['dropped'].values()), 1000)

    def test_writer_ignores_group_stop_signal_and_flushes(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {}, None, 'observe')
            try:
                self.wait_writer(recorder)
                os.kill(recorder.process.pid, signal.SIGINT)
                recorder.record('stop', reason='KeyboardInterrupt')
            finally:
                recorder.close()
            summary = json.loads((recorder.path / 'summary.json').read_text())
            self.assertTrue(summary['flush_complete'])
            self.assertEqual(summary['written']['stop'], 1)

    def test_dead_writer_does_not_hang_or_claim_complete(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {}, None, 'observe', close_timeout_s=.2)
            self.wait_writer(recorder)
            recorder.process.kill()
            recorder.process.join(timeout=2)
            recorder.record('observation', request_id=1, payload={})
            recorder.close()
            recorder.close()  # Closing twice must not touch released IPC objects.
            summary = json.loads((recorder.path / 'summary.json').read_text())
            self.assertFalse(summary['flush_complete'])
            self.assertFalse(summary['written_counts_known'])
            self.assertIsNone(summary['unwritten_queue_size'])
            self.assertEqual(summary['dropped']['observation'], 1)

    def test_stalled_writer_close_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {}, None, 'observe', queue_size=1, close_timeout_s=.2)
            self.wait_writer(recorder)
            os.kill(recorder.process.pid, signal.SIGSTOP)
            recorder.record('observation', request_id=1, payload={})
            started = time.monotonic()
            recorder.close()
            self.assertLess(time.monotonic() - started, 2)
            self.assertFalse(recorder.process.is_alive())
            self.assertFalse(json.loads((recorder.path / 'summary.json').read_text())['flush_complete'])

    def test_serialization_error_reports_unwritten_events(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {}, None, 'observe')
            recorder.record('bad', value=float('nan'))
            recorder.close()
            summary = json.loads((recorder.path / 'summary.json').read_text())
            self.assertFalse(summary['flush_complete'])
            self.assertTrue(summary['written_counts_known'])
            self.assertIn('ValueError', summary['writer_error'])
            self.assertEqual(summary['unwritten_queue_size'], 1)

    def test_large_payload_is_flushed_before_fifo_sentinel(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {}, None, 'observe')
            payload = {'images': {'head': 'x' * 2_000_000}}
            for i in range(4):
                recorder.record('observation', request_id=i, payload=payload)
            recorder.close()
            summary = json.loads((recorder.path / 'summary.json').read_text())
            self.assertTrue(summary['flush_complete'])
            self.assertEqual(summary['written']['observation'], 4)
            self.assertEqual(json.loads((recorder.path / 'requests/000003.json').read_text()), payload)

    def test_disk_error_is_reported_without_blocking_control(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = RolloutRecorder(root, {}, None, 'observe')
            self.wait_writer(recorder)
            (recorder.path / 'requests').rmdir()
            recorder.record('observation', request_id=1, payload={})
            recorder.close()
            summary = json.loads((recorder.path / 'summary.json').read_text())
            self.assertFalse(summary['flush_complete'])
            self.assertIn('FileNotFoundError', summary['writer_error'])
            self.assertEqual(summary['unwritten_queue_size'], 1)


if __name__ == '__main__':
    unittest.main()
