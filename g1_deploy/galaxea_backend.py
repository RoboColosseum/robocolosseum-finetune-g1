"""G05 G1 adapter using the upstream training processor and confirmed layout."""
import json
from pathlib import Path
import threading
import time
import numpy as np
from .catalog import ROOT, model_spec
from .policy_server import GrootBackend, decode_observation
from .lingbot_backend import restore_constant_actions

ACTION_PARTS = [('pivot', 0, 7), ('left_ee_pose', 7, 6), ('right_ee_pose', 13, 6),
                ('left_hand', 19, 2), ('right_hand', 21, 2)]
STATE_PARTS = [('left_arm', 15, 7), ('right_arm', 22, 7), ('waist', 12, 3)]


def load_config(path):
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.tokenizer.utils.parts_meta_utils import compute_grouped_dims
    register_default_resolvers()
    source = ROOT / 'third_party/GalaxeaVLA'
    # Resolve checkpoint-relative training config references without changing cwd.
    text = (path / 'config.yaml').read_text().replace(
        'configs/data/parts_meta/g1.yaml', str(ROOT / 'config/galaxea/g1.yaml'))
    text = text.replace('configs/data/_transforms.yaml', str(source / 'configs/data/_transforms.yaml'))
    cfg = OmegaConf.create(text)
    cfg.model.model_arch.hf_processor_path = str(ROOT / 'models/base/G05/qwen3_5_2b_base_processor')
    cfg.tokenizer.vq_config.ckpt_dir = str(path / 'action_tokenizer.pt')
    cfg.ckpt_path = str(path / 'model.pt')
    cfg.run_dir = str(path)
    layout = OmegaConf.load(ROOT / 'config/galaxea/g1.yaml')
    dims = compute_grouped_dims(layout.parts_meta, layout.merge_spec)
    if list(dims.items()) != list(cfg.tokenizer.vq_config.parts_meta.items()) or sum(dims.values()) != 27:
        raise ValueError('G05 tokenizer layout differs from confirmed training layout')
    shape = cfg.data.processors.unitree_g1.shape_meta
    for name, expected in [('action', ACTION_PARTS), ('state', STATE_PARTS)]:
        actual = [(m.key, m.start_index, m.raw_shape) for m in shape[name]]
        if actual != expected or any(m.shape != m.raw_shape for m in shape[name]):
            raise ValueError('G05 ' + name + ' mapping differs from G1 contract')
    if cfg.data.action_size != 32 or cfg.model.model_arch._target_ != 'g05.models.g05.g05_policy_qwen35.G05PolicyQwen35':
        raise ValueError('Unexpected G05 architecture or horizon')
    return cfg


class GalaxeaBackend(GrootBackend):
    def __init__(self, key, device):
        import torch
        from hydra.utils import instantiate
        from g05.utils.checkpoint.checkpoint_utils import _init_empty_weights_safe
        from g05.utils.data.processor_utils import build_processors
        from g05.utils.data.normalizer import load_dataset_stats_from_json
        from g05.data_processor.transforms.action_filter import BaseActionFilter
        from g05.models.g05.inferencer import PolicyInferencer
        self.spec = model_spec(key)
        self.profile, self.length, self.head_mode = key, 32, 'stereo_full'
        self.lock = threading.Lock()
        self.action_order = list(range(23))
        self.action_order[3], self.action_order[5] = 5, 3
        path = Path(self.spec['model_path'])
        cfg = load_config(path)
        print('Building G05 with strict checkpoint loading', flush=True)
        with _init_empty_weights_safe() as deferred:
            policy = instantiate(cfg.model.model_arch)
        checkpoint = torch.load(path / 'model.pt', mmap=True, map_location='cpu', weights_only=False)
        state = checkpoint['model_state_dict']
        for name, value in state.items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('Nonfinite G05 checkpoint tensor: ' + name)
        policy.load_state_dict(state, strict=True, assign=True)
        for module, weights, strict, kwargs in deferred:
            if any(t.is_meta for t in list(module.parameters()) + list(module.buffers())):
                module.load_state_dict(weights, strict=True, assign=True, **kwargs)
        if any(t.is_meta for t in list(policy.parameters()) + list(policy.buffers())):
            raise ValueError('Unloaded G05 meta tensors')
        # Match upstream serving: retain FP32 weights when configured, use BF16 autocast.
        if cfg.model.get('model_weights_to_bf16', True):
            policy = policy.to(torch.bfloat16)
        policy.apply_fp32_params()
        self.policy = policy.to(device).eval()
        self.policy.action_tokenizer.to(device)
        self.processor = build_processors(cfg)
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(path / 'dataset_stats.json'))
        self.processor.eval()
        proc = self.processor['unitree_g1']
        proc.action_filter = BaseActionFilter()
        proc.action_filter.set_shape_meta(proc.shape_meta)
        proc.action_horizon = self.length
        self.inferencer = PolicyInferencer(self.policy, self.processor, device=device)
        stats = json.loads((path / 'dataset_stats.json').read_text())['unitree_g1']['action']
        self.minimum = np.concatenate([stats[k]['global_min'] for k, _, _ in ACTION_PARTS])[self.action_order]
        self.maximum = np.concatenate([stats[k]['global_max'] for k, _, _ in ACTION_PARTS])[self.action_order]
        if self.minimum.shape != (23,) or not np.isfinite([self.minimum, self.maximum]).all():
            raise ValueError('Invalid G05 bounds')

    def schema(self):
        result = super().schema()
        result.update(model_key=self.profile, family='galaxea', task=self.spec['task'],
                      model_path=self.spec['model_path'], weight_files=self.spec['weight_files'],
                      mapping_status='User-confirmed training G1 layout; pivot RPY converted to WBC YPR; physical semantics unverified')
        result['constant_action_dimensions'] = np.flatnonzero(self.minimum == self.maximum).tolist()
        result['action_output'] = 'finite postprocessed actions with exact training constants restored; raw_model_actions retained'
        return result

    def infer(self, payload):
        if payload.get('task') != self.spec['task']:
            raise ValueError('Task differs from selected G05 checkpoint')
        obs = decode_observation(payload, self.head_mode)
        raw = {'images': {k: np.ascontiguousarray(obs['observation.images.' + k].transpose(2, 0, 1))
                          for k in ('head', 'left_wrist', 'right_wrist')},
               'state': {k: obs['observation.state'][start:start + size] for k, start, size in STATE_PARTS},
               'task': self.spec['task'], 'frequency': 30, 'embodiment_type': 'unitree_g1'}
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('Another inference is already running')
        try:
            started = time.monotonic()
            output = self.inferencer.infer_one(build_observation(raw, self.processor))
            values = []
            for name, _, size in ACTION_PARTS:
                part = output[name].detach().float().cpu().numpy()
                if part.shape == (1, self.length, size):
                    part = part[0]
                if part.shape != (self.length, size):
                    raise ValueError(f'Unexpected G05 output shape for {name}: {part.shape}')
                values.append(part)
            actions = self.validate_actions(np.concatenate(values, axis=-1)[:, self.action_order])
            restored, fixed = restore_constant_actions(actions, self.minimum, self.maximum)
            return {'actions': restored.tolist(), 'raw_model_actions': actions.tolist(),
                    'constant_action_dimensions': fixed,
                    'constant_values_restored': int(np.count_nonzero(restored != actions)),
                    'inference_s': time.monotonic() - started}
        finally:
            self.lock.release()


def build_observation(raw, processor):
    """Mirror upstream serve_policy.build_obs_dict, without WebSocket imports."""
    import torch
    proc = processor['unitree_g1']
    steps = proc.num_obs_steps
    return {
        'images': {k: torch.from_numpy(v.copy()).unsqueeze(0).expand(steps, -1, -1, -1)
                   for k, v in raw['images'].items()},
        'state': {k: torch.from_numpy(v.copy()).unsqueeze(0).expand(steps, -1).float()
                  for k, v in raw['state'].items()},
        'task': raw['task'], 'frequency': raw['frequency'], 'embodiment': 'unitree_g1',
        'action': {m['key']: torch.zeros(proc.action_horizon, m['raw_shape'])
                   for m in proc.shape_meta['action']},
        'action_is_pad': torch.ones(proc.action_horizon, dtype=torch.bool),
        'state_is_pad': torch.zeros(steps, dtype=torch.bool),
        'image_is_pad': torch.zeros(steps, dtype=torch.bool), 'idx': 0,
    }
