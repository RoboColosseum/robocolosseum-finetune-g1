"""Checkpoint-specific LeRobot adapters behind the common G1 HTTP contract."""
import argparse
import dataclasses
import fcntl
import json
import logging
from pathlib import Path
import tempfile
import threading

from .catalog import ROOT, model_spec
from .core import CAMERAS, read_stats
from .policy_server import GrootBackend, make_server

LOG = logging.getLogger(__name__)


def compatible_config(raw, fields):
    """Translate only audited version differences; reject unknown active options."""
    c = dict(raw)
    family = c['type']
    if family == 'groot' and 'dtype' not in fields and 'dtype' in c:
        if c.pop('dtype') != 'float32':
            raise ValueError('Expected float32 GR00T checkpoint')
        c['model_params_fp32'] = True
    if family == 'pi05':
        if 'use_visual_memory' not in fields:
            if c.get('use_visual_memory', False) or c.get('use_proprioceptive_memory', False):
                raise ValueError('This LeRobot version cannot run memory-enabled PI05')
            for key in ('use_visual_memory', 'use_proprioceptive_memory', 'memory_frames',
                        'memory_stride', 'memory_temporal_attention_every'):
                c.pop(key, None)
        if 'rtc_training_max_delay' not in fields:
            if c.pop('rtc_training_max_delay', 0) != 0:
                raise ValueError('Unsupported PI05 RTC training configuration')
        if 'text_tokenizer_name' not in fields:
            if c.pop('text_tokenizer_name', 'google/paligemma-3b-pt-224') != 'google/paligemma-3b-pt-224':
                raise ValueError('Unexpected PI05 tokenizer')
    if family == 'molmoact2':
        if 'dtype' not in fields and 'dtype' in c:
            c['model_dtype'] = c.pop('dtype')
        if 'train_mode_vlm' not in fields and 'train_mode_vlm' in c:
            mode = c.pop('train_mode_vlm')
            if mode not in ('lora', 'full'):
                raise ValueError(f'Unsupported Molmo VLM mode: {mode}')
            c['enable_lora_vlm'] = mode == 'lora'
        if 'norm_stats_path' not in fields:
            if c.pop('norm_stats_path', None) is not None:
                raise ValueError('External Molmo normalization statistics are not supported')
        if 'compile_model' not in fields:
            if c.pop('compile_model', False):
                raise ValueError('Unsupported compiled Molmo checkpoint')
        if 'llm_residual_dropout' not in fields:
            # Evaluation disables dropout; this training-only scalar has no tensors.
            c.pop('llm_residual_dropout', None)
    unknown = set(c) - fields - {'type'}
    if unknown:
        raise ValueError(f'Unsupported config fields: {sorted(unknown)}')
    return c


def load_config_and_processors(spec, device, *, processors=True):
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    cls = get_policy_class(spec['family'])
    cfg_cls = cls.config_class
    path = Path(spec['model_path'])
    raw = json.loads((path / 'config.json').read_text())
    if raw['type'] != spec['family'] or raw['chunk_size'] != spec['chunk_size']:
        raise ValueError('Checkpoint family/horizon differs from model catalog')
    if raw.get('use_relative_actions', False):
        raise ValueError('Relative actions require a separate state/action adapter')
    if spec['family'] == 'molmoact2':
        expected_actions = ['pivot_vx', 'pivot_vy', 'pivot_vyaw', 'pivot_roll', 'pivot_pitch', 'pivot_yaw', 'pivot_height']
        expected_actions += [f'{side}_ee_{axis}' for side in ('left', 'right') for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')]
        expected_actions += ['left_trigger', 'left_squeeze', 'right_trigger', 'right_squeeze']
        if raw.get('dataset_feature_names', {}).get('action') != expected_actions:
            raise ValueError('MolmoAct2 action names/order differ from the G1 WBC mapping')
        if raw['image_keys'] != [f'observation.images.{name}' for name in CAMERAS]:
            raise ValueError('Unexpected MolmoAct2 camera order')
    if raw['output_features']['action']['shape'] != [23]:
        raise ValueError('Expected 23-D WBC action')
    expected = {f'observation.images.{name}': [3, 480, 1280 if name == 'head' else 640] for name in CAMERAS}
    expected.update({'observation.state': [29], 'observation.state.wbc': [23]})
    if {k:v['shape'] for k,v in raw['input_features'].items()} != expected:
        raise ValueError('Checkpoint observation schema differs from G1 adapter')
    c = compatible_config(raw, {f.name for f in dataclasses.fields(cfg_cls)})
    c['device'] = device
    with tempfile.TemporaryDirectory(prefix='g01-policy-config-') as tmp:
        (Path(tmp) / 'config.json').write_text(json.dumps(c))
        cfg = cfg_cls.from_pretrained(tmp)
    # Relative training-host path in saved config must not be used by any loader.
    cfg.pretrained_path = path
    if not processors:
        return cls, cfg, None, None
    overrides = {'device_processor': {'device': device}}
    if spec['family'] == 'groot':
        overrides['groot_n1_7_vlm_encode_v1'] = {'device': device}
    elif spec['family'] == 'pi05':
        overrides['tokenizer_processor'] = {'tokenizer_name': str(path / 'tokenizer')}
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path), preprocessor_overrides=overrides)
    if spec['family'] == 'groot':
        from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep
        for step in pre.steps:
            if isinstance(step, GrootN17VLMEncodeStep):
                _ = step.proc
    return cls, cfg, pre, post


class UniversalBackend(GrootBackend):
    def __init__(self, key, device):
        import torch
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: run nvidia-smi and repair the NVIDIA kernel driver first')
        self.spec = model_spec(key)
        self.torch = torch
        self.profile = key
        self.length = self.spec['chunk_size']
        self.head_mode = self.spec['head_mode']
        self.action_order = list(range(23))
        self.action_order[3], self.action_order[5] = 5, 3
        cls, cfg, self.pre, self.post = load_config_and_processors(self.spec, device)
        # Fail on weight key/shape incompatibility rather than silently keeping
        # base-model or randomly initialized parameters from another API revision.
        if self.spec['family'] == 'molmoact2':
            from .molmoact2_backend import load_molmoact2
            self.policy = load_molmoact2(cls, cfg, self.spec['model_path'])
        else:
            self.policy = cls.from_pretrained(self.spec['model_path'], config=cfg, strict=True).to(device).eval()
        self.policy.reset()
        post_cfg = json.loads((Path(self.spec['model_path']) / 'policy_postprocessor.json').read_text())
        stats_files = [s['state_file'] for s in post_cfg['steps'] if 'state_file' in s]
        if len(stats_files) != 1:
            raise ValueError('Expected one checkpoint action normalization state file')
        stats = read_stats(Path(self.spec['model_path']) / stats_files[0])
        self.minimum = stats['action.min'][self.action_order]
        self.maximum = stats['action.max'][self.action_order]
        if 'ang_z_limit' in self.spec:
            self.minimum[2] = -self.spec['ang_z_limit']
            self.maximum[2] = self.spec['ang_z_limit']
        self.lock = threading.Lock()

    def schema(self):
        s = super().schema()
        s.update(model_key=self.spec['model_key'], family=self.spec['family'],
                 task=self.spec['task'], weight_sha256=self.spec['weight_sha256'],
                 model_path=self.spec['model_path'],
                 mapping_status='dataset metadata RPY converted to WBC YPR; physical semantics unverified')
        if 'ang_z_limit' in self.spec:
            s['action_bound_overrides'] = {'ang_z': [-self.spec['ang_z_limit'], self.spec['ang_z_limit']]}
        return s

    def image_tensor(self, value):
        # These 50k GR00T checkpoints trained with every camera resized to
        # 640x480 (including squashed stereo head). Preserve full head images
        # on the wire/in recordings, then reproduce that transform here.
        if self.spec['family'] == 'groot' and self.head_mode == 'stereo_full':
            import cv2
            if value.shape[:2] != (480, 640):
                value = cv2.resize(value, (640, 480), interpolation=cv2.INTER_LINEAR)
        return super().image_tensor(value)

    def infer(self, payload):
        if payload.get('task') != self.spec['task']:
            raise ValueError(f"This checkpoint expects task {self.spec['task']!r}")
        return super().infer(payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-key', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8008)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    (ROOT / 'outputs').mkdir(exist_ok=True)
    gpu_lock = (ROOT / 'outputs/policy-model.lock').open('a')
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another common-framework policy is already loading/running')
    backend = make_backend(a.model_key, a.device)
    server = make_server((a.host, a.port), backend)
    LOG.info('Ready: %s on %s:%d', a.model_key, a.host, a.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def make_backend(key, device):
    spec = model_spec(key)
    if spec.get('unavailable_reason'):
        raise ValueError(spec['unavailable_reason'])
    if spec['family'] == 'lingbot':
        from .lingbot_backend import LingbotBackend
        return LingbotBackend(key, device)
    if spec['family'] == 'galaxea':
        from .galaxea_backend import GalaxeaBackend
        return GalaxeaBackend(key, device)
    return UniversalBackend(key, device)


if __name__ == '__main__':
    main()
