"""Bounded process-based rollout writer. Never imports DDS or sends commands."""
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import multiprocessing
import os
from pathlib import Path
import queue
import re
import signal
import time
import uuid

LOG = logging.getLogger(__name__)


def _write_events(path, events, result, parent_pid):
    # The UI sends SIGINT to the entire client process group. Let the parent
    # stop control and send the FIFO sentinel; do not interrupt the disk flush.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    written = Counter()
    error = None
    drained = False
    try:
        with (Path(path) / 'events.jsonl').open('w', buffering=1) as stream:
            while True:
                try:
                    event = events.get(timeout=.25)
                except queue.Empty:
                    if os.getppid() != parent_pid:
                        raise RuntimeError('Recorder parent exited before closing the episode')
                    continue
                if event is None:
                    drained = True
                    break
                if 'payload' in event:
                    relative = f"requests/{event['request_id']:06d}.json"
                    (Path(path) / relative).write_text(json.dumps(event.pop('payload'), allow_nan=False) + '\n')
                    event['payload_file'] = relative
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                written[event['event']] += 1
    except Exception as exc:
        error = repr(exc)
    finally:
        try:
            result.send(dict(written=dict(written), writer_error=error,
                             drained=drained, writer_pid=os.getpid()))
        except (BrokenPipeError, OSError):
            pass
        result.close()


def rollout_identity(config, schema=None):
    schema = schema or {}
    task = config.get('task', schema.get('task')) or 'unknown-task'
    task = {'carrying box': 'carry-box'}.get(task, task)
    model = (config.get('model_key') or schema.get('model_key')
             or Path(config.get('model_path', '')).name or schema.get('profile') or 'unknown-model')
    def slug(value):
        return re.sub(r'[^a-zA-Z0-9_.-]+', '-', str(value)).strip('.-')[:100] or 'unknown'
    return slug(task), slug(model)


class RolloutRecorder:
    def __init__(self, root, config, schema, mode, queue_size=256, close_timeout_s=8):
        if queue_size < 1 or not 0 < close_timeout_s <= 15:
            raise ValueError('Recorder requires a bounded queue and a close timeout in (0, 15] seconds')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        task_name, model_name = rollout_identity(config, schema)
        self.path = Path(root) / f'{task_name}__{model_name}__{stamp}-{mode}-{uuid.uuid4().hex[:6]}'
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'requests').mkdir()
        # Spawn instead of fork: never inherit live DDS/camera threads or locks.
        ctx = multiprocessing.get_context('spawn')
        self.queue = ctx.Queue(maxsize=queue_size)
        # A failed writer must not leave the client waiting for its IPC feeder.
        self.queue.cancel_join_thread()
        self._result, sender = ctx.Pipe(duplex=False)
        self.stopping = False
        self._closed = False
        self.close_timeout_s = close_timeout_s
        self.written = Counter()
        self.accepted = Counter()
        self.dropped = Counter()
        self.error = None
        self.started = time.time()
        manifest = {
            'format': 'g01-rollout-v1', 'mode': mode, 'start_unix_s': self.started,
            'writer_backend': 'process', 'writer_queue_size': queue_size,
            'task_name': task_name, 'model_name': model_name,
            'config': config, 'policy_schema': schema,
            'image_encoding': 'base64 JPEG RGB decoded with OpenCV BGR-to-RGB conversion',
            'image_scope': 'exact policy request images after configured client camera transform; not continuous raw camera video',
            'sampling': 'policy inputs per inference request; state and selected action per control attempt',
            'command_semantics': 'command_submitted is accepted by client transport, not DDS delivery or physical tracking proof',
            'wbc_state_semantics': 'commanded base + measured arm FK + last gripper controls, not all measured',
            'timestamps': 'wall_unix_ns plus host monotonic_s; state/camera timestamps are host receive times',
        }
        (self.path / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        self.process = ctx.Process(target=_write_events,
            args=(str(self.path), self.queue, sender, os.getpid()),
            daemon=True, name='rollout-writer')
        self.process.start()
        sender.close()
        LOG.info('Saving rollout to %s', self.path.resolve())

    def record(self, kind, **data):
        event = {'event': kind, 'wall_unix_ns': time.time_ns(), 'monotonic_s': time.monotonic(), **data}
        if self.error or self.stopping or not self.process.is_alive():
            self.dropped[kind] += 1
            return
        try:
            self.queue.put_nowait(event)
            self.accepted[kind] += 1
        except queue.Full:
            self.dropped[kind] += 1

    def close(self):
        if self._closed:
            return
        self.stopping = True
        deadline = time.monotonic() + self.close_timeout_s
        # The FIFO sentinel follows all accepted events, including events still
        # being serialized by multiprocessing's background IPC feeder.
        while self.process.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(None, timeout=min(.1, max(0, deadline - time.monotonic())))
                break
            except queue.Full:
                continue
        outcome = None
        while time.monotonic() < deadline:
            if self._result.poll(min(.05, max(0, deadline - time.monotonic()))):
                try:
                    outcome = self._result.recv()
                except EOFError:
                    pass
                break
            if not self.process.is_alive():
                break
        self.process.join(timeout=max(0, deadline - time.monotonic()))
        # The child can send its result between the last poll and its exit.
        if outcome is None and self._result.poll():
            try:
                outcome = self._result.recv()
            except EOFError:
                pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=.5)
        if outcome is not None:
            self.written.update(outcome['written'])
            self.error = outcome['writer_error']
        else:
            self.error = f'Writer did not report completion (exit code {self.process.exitcode}); counts unknown'
        unwritten = max(0, sum(self.accepted.values()) - sum(self.written.values())) if outcome else None
        summary = {'end_unix_s': time.time(), 'written': dict(self.written),
                   'dropped': dict(self.dropped), 'writer_error': self.error,
                   'accepted': dict(self.accepted), 'writer_backend': 'process',
                   'writer_pid': self.process.pid, 'written_counts_known': outcome is not None,
                   'unwritten_queue_size': unwritten,
                   'flush_complete': bool(outcome and outcome['drained'] and not self.error
                                          and unwritten == 0 and self.process.exitcode == 0)}
        self.queue.close()
        self._result.close()
        self._closed = True
        try:
            (self.path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        except OSError:
            LOG.exception('Cannot save rollout summary')
        LOG.info('Rollout saved: %s; summary=%s', self.path.resolve(), summary)
