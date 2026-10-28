"""GPU load and synthetic inference checks; no robot imports or DDS writes."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from g1_deploy.catalog import catalog, model_spec
    from g1_deploy.multi_policy import make_backend
    from g1_deploy.wbc_client import encode_payload, validate_schema
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', default='outputs/model-smoke-results.json')
    p.add_argument('--observation', help='Replay a measured NPZ observation without opening DDS')
    args = p.parse_args()
    (ROOT / 'outputs').mkdir(exist_ok=True)
    gpu_lock = (ROOT / 'outputs/policy-model.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    keys = args.models or list(catalog()['models'])
    results = {'validation': 'synthetic images/state; loading and inference only, no DDS or task-success evidence', 'models': {}}
    if args.observation:
        results['validation'] = 'replayed measured observation; no DDS, actuation or task-success evidence'
        results['observation'] = args.observation
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for key in keys:
        spec = model_spec(key)
        print('Testing', key, flush=True)
        backend = None
        started = time.monotonic()
        try:
            backend = make_backend(key, args.device)
            schema = backend.schema()
            validate_schema(schema)
            width = 1280 if spec['head_mode'] == 'stereo_full' else 640
            images = {'head': np.full((480, width, 3), 127, np.uint8),
                      'left_wrist': np.full((480, 640, 3), 127, np.uint8),
                      'right_wrist': np.full((480, 640, 3), 127, np.uint8)}
            payload = encode_payload(images, np.zeros(29), np.zeros(23), spec['task'])
            if args.observation:
                with np.load(args.observation, allow_pickle=False) as observation:
                    payload = encode_payload({name: observation[name] for name in images},
                                             observation['state'], observation.get('wbc_state'), spec['task'])
            responses = [backend.infer(payload) for _ in range(2)]
            results['models'][key] = {'ok': True, 'schema': schema, 'load_and_test_s': time.monotonic()-started,
                                     'inference_s': [r['inference_s'] for r in responses],
                                     'shape': list(np.asarray(responses[-1]['actions']).shape),
                                     'responses': responses}
        except Exception as exc:
            traceback.print_exc()
            results['models'][key] = {'ok': False, 'error': repr(exc), 'elapsed_s': time.monotonic()-started}
        finally:
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            output.write_text(json.dumps(results, indent=2) + '\n')
        print(key, {k: v for k, v in results['models'][key].items() if k not in ('responses', 'schema')}, flush=True)
    if not all(r['ok'] for r in results['models'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
