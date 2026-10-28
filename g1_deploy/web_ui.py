"""Small UI for the existing model runner; stdlib only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .catalog import ROOT, catalog, robot_config, model_downloaded
from .rollout_store import inventory, delete_rollout, recording_processes

def validate_request(data, models, interfaces, mode='execute'):
    key = data.get('model')
    if key not in models:
        raise ValueError('Select a model from the catalog')
    if mode not in ('execute', 'policy', 'test') or data.get('mode', mode) != mode:
        raise ValueError('This action does not support the requested mode; refresh the page')
    interface = ''
    if mode == 'execute':
        interface = robot_config(key, os.environ.get('ROBOT_CONFIG'))['interface']
        if interface not in interfaces:
            raise ValueError('Configured robot network interface not found: ' + interface)
    return key, mode, interface, 0


class Runner:
    def __init__(self, mode='execute'):
        self.mode = mode
        self.lock = threading.RLock()
        self.proc = None
        self.group = None
        self.stopping = False
        self.info = {}

    def group_alive(self):
        if self.group is None:
            return False
        if self.proc is not None:
            self.proc.poll()  # reap the group leader before probing
        try:
            os.killpg(self.group, 0)
            return True
        except ProcessLookupError:
            return False

    def status(self):
        with self.lock:
            active = self.group_alive()
            code = self.proc.poll() if self.proc else None
            state = 'stopping' if self.stopping else ('running' if active else ('idle' if not self.proc else ('finished' if code == 0 else 'stopped' if code in (130, -2) else 'failed')))
            result = dict(self.info, state=state, active=active, exit_code=code)
            result['log'] = ''
            if self.info.get('log_path'):
                with Path(self.info['log_path']).open('rb') as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 24000))
                    result['log'] = f.read().decode('utf-8', errors='replace')
                for line in result['log'].splitlines():
                    if line.startswith('Loading ') and '; log: ' in line:
                        policy_log = Path(line.split('; log: ', 1)[1]).resolve()
                        if policy_log.is_relative_to((ROOT / 'outputs/runs').resolve()) and policy_log.is_file():
                            with policy_log.open('rb') as f:
                                f.seek(0, 2)
                                f.seek(max(0, f.tell() - 16000))
                                result['log'] += '\n--- Policy server log ---\n' + f.read().decode('utf-8', errors='replace')
                        break
            return result

    def start(self, data):
        with self.lock:
            if self.group_alive() or self.stopping:
                raise ValueError('This process is already running; stop it first')
            models = catalog()['models']
            key, mode, interface, duration = validate_request(data, models, [n for _, n in socket.if_nameindex()], self.mode)
            if models[key].get('unavailable_reason'):
                raise ValueError(models[key]['unavailable_reason'])
            if not model_downloaded(models[key], ROOT):
                raise ValueError('Model download is incomplete')
            path = ROOT / 'outputs/ui' / (time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.log')
            path.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            if mode == 'execute':
                env['G1_REQUIRE_POLICY'] = '1'
            args = ['bash', str(ROOT / 'scripts/run_model.sh'), key, mode, interface, str(duration)]
            if mode == 'test':
                args = [str(ROOT / '.venv-policy/bin/python'), str(ROOT / 'scripts/test_policy_replay.py'), '--model', key]
            with path.open('wb') as log:
                proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.proc = proc
            self.group = proc.pid
            self.info = dict(model=key, mode=mode, interface=interface, duration=duration, pid=proc.pid, log_path=str(path), started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            return self.status()

    def stop(self):
        with self.lock:
            if not self.group_alive() or self.stopping:
                return self.status()
            self.stopping = True
            group = self.group
            os.killpg(group, signal.SIGINT)
            threading.Thread(target=self._finish_stop, args=(group,), daemon=True).start()
            return self.status()

    def _finish_stop(self, group):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with self.lock:
                if not self.group_alive():
                    break
            time.sleep(.2)
        with self.lock:
            if self.group == group and self.group_alive():
                try:
                    os.killpg(group, signal.SIGKILL)
                    self.info['forced_stop'] = True
                except ProcessLookupError:
                    pass
            if self.proc:
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            self.stopping = False

    def close(self):
        self.stop()
        while self.stopping:
            time.sleep(.1)


class Console:
    def __init__(self):
        self.lock = threading.RLock()
        self.client = Runner('execute')
        self.policy = Runner('policy')
        self.test = Runner('test')

    def schema(self):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8008/schema', timeout=.5) as response:
                return json.load(response)
        except (OSError, ValueError):
            return None

    def status(self):
        with self.lock:
            schema = self.schema()
            policy = self.policy.status()
            policy['ready'] = schema is not None
            policy['loaded_model'] = schema.get('model_key', schema.get('profile')) if schema else None
            policy['external'] = schema is not None and not policy['active']
            if policy['external']:
                policy['state'] = 'external'
            elif policy['active'] and not self.policy.stopping:
                policy['state'] = 'ready' if schema else 'loading'
            return dict(client=self.client.status(), server=policy, test=self.test.status())

    def start_server(self, data):
        with self.lock:
            if self.test.group_alive():
                raise ValueError('Wait for the inference test to finish')
            if self.client.group_alive():
                raise ValueError('Stop the client before starting or switching the server')
            with socket.socket() as probe:
                probe.settimeout(.5)
                occupied = probe.connect_ex(('127.0.0.1', 8008)) == 0
            if occupied:
                raise ValueError('A policy server is already listening on port 8008; stop it before switching')
            self.policy.start(data)
            return self.status()

    def execute(self, data):
        with self.lock:
            if self.test.group_alive():
                raise ValueError('Wait for the inference test to finish')
            if self.policy.stopping:
                raise ValueError('The server is stopping')
            schema = self.schema()
            if not schema or schema.get('model_key') != data.get('model'):
                raise ValueError('Start the selected model server and wait until it is Ready before Execute')
            self.client.start(data)
            return self.status()

    def test_policy(self, data):
        with self.lock:
            if self.client.group_alive() or self.client.stopping or self.policy.stopping:
                raise ValueError('Stop the robot client and wait for a ready server before testing')
            schema = self.schema()
            if not schema or schema.get('model_key') != data.get('model'):
                raise ValueError('Start the selected model server before testing')
            self.test.start(data)
            return self.status()

    def stop_client(self):
        with self.lock:
            self.client.stop()
            return self.status()

    def rollout_list(self):
        rows = inventory(ROOT / 'outputs/rollouts')
        with self.lock:
            blocked = self.client.group_alive() or self.client.stopping or bool(recording_processes())
        return dict(rollouts=rows, deletion_blocked=bool(blocked))

    def remove_rollout(self, data):
        # Serialize against Execute so the UI cannot start a recording mid-delete.
        with self.lock:
            if self.client.group_alive() or self.client.stopping or recording_processes():
                raise ValueError('Stop the robot client before deleting recordings')
            return delete_rollout(ROOT / 'outputs/rollouts', data.get('id'), data.get('confirm_name'))

    def stop_server(self):
        with self.lock:
            if self.test.group_alive():
                raise ValueError('Wait for the inference test to finish before stopping the server')
            if self.client.group_alive() or self.client.stopping:
                raise ValueError('Stop the client before stopping the server')
            self.policy.stop()
            return self.status()

    def close(self):
        self.test.close()
        self.client.close()
        self.policy.close()


def handler_for(runner):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, code, body, content_type='application/json; charset=utf-8'):
            body = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                return self.reply(200, (Path(__file__).with_name('web_ui.html')).read_text(), 'text/html; charset=utf-8')
            if self.path == '/api/catalog':
                models = []
                for key, spec in catalog()['models'].items():
                    downloaded = model_downloaded(spec, ROOT)
                    models.append(dict(key=key, family=spec['family'], task=spec['task_slug'], downloaded=downloaded,
                                       available=downloaded and not spec.get('unavailable_reason'),
                                       unavailable_reason=spec.get('unavailable_reason')))
                return self.reply(200, dict(models=models))
            if self.path == '/api/status':
                return self.reply(200, runner.status())
            if self.path == '/api/rollouts':
                try:
                    return self.reply(200, runner.rollout_list())
                except (OSError, ValueError) as exc:
                    return self.reply(503, {'error': str(exc)})
            self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            # No login token; still reject cross-site browser commands.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.reply(403, {'error': 'Cross-origin commands are not allowed'})
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                return self.reply(415, {'error': 'Use application/json'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Invalid body length')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('Invalid request')
                if self.path == '/api/server/start':
                    return self.reply(200, runner.start_server(data))
                if self.path == '/api/server/test':
                    return self.reply(200, runner.test_policy(data))
                if self.path == '/api/client/execute':
                    return self.reply(200, runner.execute(data))
                if self.path == '/api/client/stop':
                    return self.reply(200, runner.stop_client())
                if self.path == '/api/server/stop':
                    return self.reply(200, runner.stop_server())
                if self.path == '/api/rollouts/delete':
                    return self.reply(200, runner.remove_rollout(data))
                self.reply(404, {'error': 'Not found'})
            except (ValueError, TypeError) as exc:
                self.reply(400, {'error': str(exc)})
            except OSError as exc:
                self.reply(500, {'error': str(exc)})
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=7860)
    a = p.parse_args()
    out = ROOT / 'outputs/ui'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'server.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runner = Console()
    server = ThreadingHTTPServer((a.host, a.port), handler_for(runner))
    server.daemon_threads = True
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    print(f'G1 UI: http://{a.host}:{a.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runner.close()


if __name__ == '__main__':
    main()
