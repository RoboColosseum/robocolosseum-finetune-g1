"""Test the selected running HTTP policy with saved observations; no DDS publishers."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_deploy.catalog import ROOT, robot_config
from g1_deploy.core import validate_chunk
from g1_deploy.wbc_client import validate_schema


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    with urllib.request.urlopen('http://127.0.0.1:8008/schema', timeout=5) as response:
        schema = json.load(response)
    validate_schema(schema)
    if schema.get('model_key') != a.model:
        raise ValueError('Loaded model differs from selected model')
    from g1_deploy.policy_server import decode_observation
    candidates = sorted((ROOT / 'outputs/rollouts').rglob('requests/000001.json'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer same-task observations; fall back only for interface compatibility testing.
    chosen = None
    fallback = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
            decode_observation(payload, schema['head_mode'])
            if fallback is None:
                fallback = (candidate, payload)
            if payload.get('task') == schema['task']:
                chosen = (candidate, payload)
                break
        except (ValueError, KeyError, OSError):
            continue
    if chosen is None:
        chosen = fallback
    if chosen is None:
        raise ValueError('No compatible saved observation available')
    source, payload = chosen
    original_task = payload.get('task')
    payload['task'] = schema['task']
    print('Recorded task:', original_task, '; inference prompt:', schema['task'], flush=True)
    print(f'Testing {a.model} with saved observation: {source}', flush=True)
    results = []
    config = robot_config(a.model)
    print('Client request timeout:', config['request_timeout_s'], 'seconds', flush=True)
    for _ in range(2):
        request = urllib.request.Request('http://127.0.0.1:8008/infer', data=json.dumps(payload).encode(),
                                         headers={'Content-Type':'application/json'})
        requested_at = time.monotonic()
        with urllib.request.urlopen(request, timeout=config['request_timeout_s']) as response:
            result = json.load(response)
        result['http_roundtrip_s'] = time.monotonic() - requested_at
        actions = np.asarray(result['actions'])
        if list(actions.shape) != schema['action_shape'] or not np.isfinite(actions).all():
            raise ValueError('Invalid action shape or nonfinite inference output')
        try:
            bounded = validate_chunk(actions, schema['action_min'], schema['action_max'], schema['action_shape'][0],
                           overshoot_fraction=config.get('action_overshoot_fraction', 0),
                           bounds_mode=config.get('action_bounds_mode', 'reject'))
            result['client_clipped_values'] = int(np.count_nonzero(bounded != actions))
            result['client_actions'] = bounded.tolist()
            result['client_bounds_mode'] = config.get('action_bounds_mode', 'reject')
            result['client_bounds_ok'] = True
        except ValueError as exc:
            result['client_bounds_ok'] = False
            result['client_bounds_error'] = str(exc)
            print('Client would reject this recorded-observation output:', exc, flush=True)
        results.append(result)
        print(f"Finite {list(actions.shape)} actions; inference {result['inference_s']:.3f}s", flush=True)
    destination = ROOT / 'outputs/replay-tests' / (time.strftime('%Y%m%dT%H%M%S') + '-' + a.model + '.json')
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(dict(model=a.model, observation=str(source), recorded_task=original_task, schema=schema,
                                           responses=results, request_timeout_s=config['request_timeout_s'], robot_commands_sent=False), indent=2)+'\n')
    if not all(r['client_bounds_ok'] for r in results):
        print('Report:', destination)
        raise SystemExit('FAIL: finite inference, but client action bounds rejected the output. No robot commands sent.')
    print('PASS: saved-observation HTTP inference and client bounds. No robot commands sent. Physical task success untested.')
    print('Report:', destination)


if __name__ == '__main__':
    main()
