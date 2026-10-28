"""Pinned model inventory and per-model WBC configuration."""
import copy
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def catalog():
    return json.loads((ROOT / 'config/models.json').read_text())


def model_downloaded(spec, root=ROOT):
    """Quick completeness check, including every shard and processor asset."""
    path = Path(root) / spec['model_path']
    files = spec.get('required_files', spec.get('weight_files', [
        {'path': 'model.safetensors', 'size': spec['weight_bytes']}]))
    return all((path / f['path']).is_file() and
               (path / f['path']).stat().st_size == f['size'] for f in files)


def model_spec(key):
    models = catalog()['models']
    if key not in models:
        raise ValueError(f'Unknown model {key!r}; choose from: {", ".join(models)}')
    spec = dict(models[key], model_key=key)
    if 'ang_z_limit' in spec:
        limit = float(spec['ang_z_limit'])
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError('ang_z_limit must be positive and finite')
        spec['ang_z_limit'] = limit
    spec['model_path'] = str(ROOT / spec['model_path'])
    return spec


def robot_config(key, base_path=None):
    spec = model_spec(key)
    local = ROOT / 'config/robot.json'
    path = base_path or os.environ.get('ROBOT_CONFIG') or (local if local.is_file() else ROOT / 'config/robot.example.json')
    c = json.loads(Path(path).read_text())
    c = copy.deepcopy(c)
    c['interface'] = os.environ.get('G1_INTERFACE', c['interface'])
    wbc = Path(os.environ.get('WBC_REPO', c['wbc_repo'])).expanduser()
    c['wbc_repo'] = str(wbc if wbc.is_absolute() else ROOT / wbc)
    c.update(model_key=key, model_path=spec['model_path'], task=spec['task'],
             chunk_size=spec['chunk_size'], rollout_dir=str(ROOT / 'outputs/rollouts/by-model' / spec['family'] / spec['task_slug']))
    bounds_mode = spec.get('action_bounds_mode', 'reject')
    if bounds_mode not in ('reject', 'clip'):
        raise ValueError('Invalid action bounds mode')
    c['action_bounds_mode'] = bounds_mode
    if 'request_timeout_s' in spec:
        timeout = float(spec['request_timeout_s'])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('request_timeout_s must be positive and finite')
        c['request_timeout_s'] = timeout
    if 'ang_z_limit' in spec:
        c['base_min'][2] = -spec['ang_z_limit']
        c['base_max'][2] = spec['ang_z_limit']
    c['camera']['head_mode'] = spec['head_mode']
    c['camera']['columns']['head'] = [0, 1280]
    return c
