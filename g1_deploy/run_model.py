"""One entry point for model selection, policy, observe, dry-run and execution."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from .catalog import ROOT, catalog, model_spec, robot_config, model_downloaded


def get_schema():
    with urllib.request.urlopen('http://127.0.0.1:8008/schema', timeout=2) as response:
        return json.load(response)


def main():
    # A background UI may inherit SIGINT ignored from its launching shell.
    # Restore Ctrl+C handling before creating policy/client subprocesses.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', nargs='?', default='list')
    p.add_argument('mode', nargs='?', choices=['policy', 'observe', 'dry-run', 'execute', 'check'], default='dry-run')
    p.add_argument('interface', nargs='?', default=None)
    p.add_argument('duration', nargs='?', type=float, default=30)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--online', action='store_true', help='Allow downloading missing base model assets')
    a = p.parse_args()
    os.chdir(ROOT)
    if a.model == 'list':
        for key, spec in catalog()['models'].items():
            present = model_downloaded(spec)
            state = 'invalid' if spec.get('unavailable_reason') else ('downloaded' if present else 'missing')
            print(f"{key:42s} {state:10s} {spec['chunk_size']:2d} steps")
        return
    spec = model_spec(a.model)
    if spec.get('unavailable_reason') and a.mode != 'observe':
        raise SystemExit(spec['unavailable_reason'])
    if a.duration < 0:
        p.error('duration must be nonnegative')
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'), TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='0' if a.online else '1', TRANSFORMERS_OFFLINE='0' if a.online else '1')
    policy_python = os.environ.get('POLICY_PYTHON', str(ROOT / '.venv-policy/bin/python'))
    if spec['family'] == 'lingbot':
        policy_python = os.environ.get('LINGBOT_PYTHON', str(ROOT / '.venv-lingbot/bin/python'))
    if spec['family'] == 'galaxea':
        policy_python = os.environ.get('GALAXEA_PYTHON', str(ROOT / '.venv-galaxea/bin/python'))
    wbc_python = os.environ.get('WBC_PYTHON', str(ROOT / '.venv-wbc/bin/python'))
    policy_args = [policy_python, '-m', 'g1_deploy.multi_policy', '--model-key', a.model, '--device', a.device]
    if a.mode == 'check':
        if spec['family'] == 'galaxea':
            if not model_downloaded(spec):
                raise ValueError('G05 download is incomplete')
            print(f'{a.model}: checkpoint files checked; use Web inference test after starting the server')
            return
        if spec['family'] == 'lingbot':
            import yaml
            from .lingbot_backend import validate_robot_mapping
            validate_robot_mapping(yaml.safe_load((Path(spec['model_path']) / 'robot_config.yaml').read_text()))
            if not model_downloaded(spec):
                raise ValueError('LingBot download is incomplete')
            print(f'{a.model}: mapping and checkpoint files checked; inference not tested')
            return
        from .multi_policy import load_config_and_processors
        os.environ.update(env)
        load_config_and_processors(spec, 'cpu')
        print(f'{a.model}: config and processors loaded; model inference not tested')
        return
    if a.mode == 'policy':
        os.execve(policy_python, policy_args, env)
    out = ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    # Prevent simultaneous clients launched through this common entry point.
    with (out / 'robot-client.lock').open('a') as run_lock:
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another run_model client is active; stop it before starting another.')
        config = robot_config(a.model, os.environ.get('ROBOT_CONFIG'))
        config['interface'] = a.interface or config['interface']
        run_dir = out / 'runs' / f'{a.model}-{uuid.uuid4().hex[:10]}'
        run_dir.mkdir(parents=True)
        config_path = run_dir / 'config.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        client = [wbc_python, '-m', 'g1_deploy.wbc_client', '--config', str(config_path)]
        child = None
        log = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            if a.mode != 'observe':
                try:
                    schema = get_schema()
                except OSError:
                    if os.environ.get('G1_REQUIRE_POLICY') == '1':
                        raise RuntimeError('The selected policy server is unavailable; start it before Execute')
                    # An occupied but unready port must not trigger another model load.
                    with socket.socket() as probe:
                        probe.settimeout(1)
                        occupied = probe.connect_ex(('127.0.0.1', 8008)) == 0
                    if occupied:
                        raise RuntimeError('Port 8008 is occupied but schema is unavailable')
                    log = (run_dir / 'policy.log').open('w')
                    child = subprocess.Popen(policy_args, env=env, stdout=log, stderr=subprocess.STDOUT)
                    print(f'Loading {a.model}; log: {run_dir / "policy.log"}', flush=True)
                    deadline = time.monotonic() + 300
                    while True:
                        if child.poll() is not None:
                            raise RuntimeError(f'Policy exited ({child.returncode}); see {run_dir / "policy.log"}')
                        try:
                            schema = get_schema()
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise RuntimeError('Policy startup timed out')
                            time.sleep(1)
                if schema.get('model_key') != a.model:
                    raise RuntimeError(f"Server is {schema.get('model_key', schema.get('profile'))}; stop that policy before switching to {a.model}")
                observation = run_dir / 'observation.npz'
                subprocess.run(client + ['--mode', 'observe', '--duration', '2', '--save-observation', str(observation)], env=env, check=True)
                subprocess.run([wbc_python, '-m', 'g1_deploy.offline_infer', '--observation', str(observation),
                                '--config', str(config_path), '--task', spec['task'], '--output', str(run_dir / 'warmup.json')], env=env, check=True)
            subprocess.run(client + ['--mode', a.mode, '--duration', str(a.duration)], env=env, check=True)
        finally:
            # Never stop a policy that was already running before this command.
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if log:
                log.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped by operator', file=sys.stderr)
        raise SystemExit(130)
