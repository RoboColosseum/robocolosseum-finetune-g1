"""LingBot VLA V2 adapter using the checkpoint's G1 FeatureTransform."""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .catalog import ROOT, model_spec
from .policy_server import GrootBackend, decode_observation


def restore_constant_actions(actions, minimum, maximum):
    """Restore exact constants despite inconsistent nonzero saved std values.

    Variable dimensions are untouched. The caller retains the original model
    output separately, so this does not erase evidence of normalization drift.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if not np.isfinite(actions).all():
        raise ValueError('Nonfinite model actions cannot be restored')
    fixed = np.flatnonzero(np.asarray(minimum) == np.asarray(maximum))
    restored = actions.copy()
    restored[:, fixed] = np.asarray(minimum)[fixed]
    return restored, fixed.tolist()


def validate_robot_mapping(robot):
    expected = {
        'states': [('observation.state.arm.position', [(15, 22), (22, 29)]),
                   ('observation.state.waist.position', [(12, 15)]),
                   ('observation.state.hand.position', [(0, 12)])],
        'actions': [('action.base.position', [(0, 3)]),
                    ('action.waist.position', [(3, 7)]),
                    ('action.end.position', [(7, 13), (13, 19)]),
                    ('action.hand.position', [(19, 23)])],
    }
    for group, entries in expected.items():
        actual = []
        source_key = 'action' if group == 'actions' else 'observation.state'
        for entry in robot[group]:
            if len(entry) != 1:
                raise ValueError('Ambiguous G1 mapping')
            key, value = next(iter(entry.items()))
            if value.get('subtract_state', False):
                raise ValueError('Relative LingBot actions are unsupported')
            slices = []
            for origin in value['origin_keys']:
                if set(origin) != {source_key}:
                    raise ValueError('Unexpected G1 mapping source')
                slices.append((origin[source_key]['start'], origin[source_key]['end']))
            actual.append((key, slices))
        if actual != entries:
            raise ValueError(f'Unrecognized LingBot G1 {group} mapping')
    images = {k: v['origin_keys'] for item in robot['images'] for k, v in item.items()}
    if images != {'observation.images.camera_top': 'observation.images.head',
                  'observation.images.camera_wrist_left': 'observation.images.left_wrist',
                  'observation.images.camera_wrist_right': 'observation.images.right_wrist'}:
        raise ValueError('Unrecognized LingBot camera mapping')


class LingbotBackend(GrootBackend):
    def __init__(self, key, device):
        import torch
        import yaml
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import AutoConfig, Qwen3VLProcessor

        sys.path.insert(0, str(ROOT / 'third_party/lingbot-vla-v2'))
        from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
        from lingbotvla.data.vla_data.utils import FeatureTransform

        self.spec = model_spec(key)
        self.profile = key
        self.torch = torch
        self.device = device
        self.length = self.spec['chunk_size']
        self.head_mode = 'stereo_full'
        self.action_order = list(range(23))
        self.action_order[3], self.action_order[5] = 5, 3
        self.lock = threading.Lock()
        path = Path(self.spec['model_path'])
        raw = json.loads((path / 'config.json').read_text())
        if raw['architectures'] != ['LingbotVlaV2Policy'] or raw['chunk_size'] != self.length:
            raise ValueError('LingBot architecture/horizon differs from catalog')
        if raw['max_action_dim'] != 55 or raw['max_state_dim'] != 55:
            raise ValueError('Expected 55-D padded LingBot space')
        robot = yaml.safe_load((path / 'robot_config.yaml').read_text())
        validate_robot_mapping(robot)
        train = yaml.safe_load((path / 'train_config.yaml').read_text())
        self.data_config = SimpleNamespace(**train['data'])
        config = LingbotVLAV2Config(**raw)
        config.tokenizer_path = str(ROOT / 'models/base/Qwen3-VL-4B-Instruct')
        base = AutoConfig.from_pretrained(config.tokenizer_path, local_files_only=True)
        for name in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                     'num_key_value_heads', 'rms_norm_eps', 'rope_theta', 'vocab_size',
                     'max_position_embeddings', 'hidden_act', 'tie_word_embeddings'):
            if hasattr(base.text_config, name):
                setattr(config, name, getattr(base.text_config, name))
        config.vision_config = base.vision_config
        config.attention_implementation = 'eager'
        config.vit_attn_implementation = 'sdpa'
        config.use_cache = True
        config.use_compile = False
        apply_lingbot_qwen3_vl_patch()
        self.processor = Qwen3VLProcessor.from_pretrained(str(path), local_files_only=True)
        self.transform = FeatureTransform(str(path / 'robot_config.yaml'), self.data_config,
                                          config, self.processor, chunk_size=self.length,
                                          norm_stats_path=str(path / 'norm_stats.json'))
        print('Building LingBot V2 and loading checkpoint shards strictly', flush=True)
        with init_empty_weights():
            policy = LingbotVlaV2Policy(config, eval=True)
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        state = {}
        for shard in sorted(set(index['weight_map'].values())):
            if Path(shard).name != shard:
                raise ValueError('Invalid checkpoint shard path')
            tensors = load_file(str(path / shard), device='cpu')
            for name, tensor in tensors.items():
                if name in state or index['weight_map'].get(name) != shard:
                    raise ValueError('Checkpoint index disagrees with shard keys')
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f'Checkpoint contains NaN/Inf weights: {shard}: {name}; obtain a corrected checkpoint')
                state[name] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
            del tensors
        if set(state) != set(index['weight_map']):
            raise ValueError('Incomplete LingBot checkpoint')
        policy.load_state_dict(state, strict=True, assign=True)
        del state
        self.policy = policy.to(device=device, dtype=torch.bfloat16).eval()
        self.policy.model._use_compile_predict_velocity = False
        self.policy.model._compiled_predict_velocity = None
        stats = json.loads((path / 'norm_stats.json').read_text())['norm_stats']
        groups = ['action.base.position', 'action.waist.position', 'action.end.position', 'action.hand.position']
        self.minimum = np.concatenate([stats[g]['min'] for g in groups])[self.action_order]
        self.maximum = np.concatenate([stats[g]['max'] for g in groups])[self.action_order]
        if self.minimum.shape != (23,) or not np.isfinite([self.minimum, self.maximum]).all():
            raise ValueError('Invalid LingBot action bounds')

    def schema(self):
        result = super().schema()
        result.update(model_key=self.spec['model_key'], family='lingbot', task=self.spec['task'],
                      model_path=self.spec['model_path'], weight_files=self.spec['weight_files'],
                      mapping_status='checkpoint robot_config maps 29-D joints and 23-D actions; RPY converted to WBC YPR; physical semantics unverified')
        result['constant_action_dimensions'] = np.flatnonzero(self.minimum == self.maximum).tolist()
        result['action_output'] = 'finite postprocessed actions with exact training constants restored; raw_model_actions retained'
        return result

    def infer(self, payload):
        from torchvision.transforms.v2 import Resize
        if payload.get('task') != self.spec['task']:
            raise ValueError(f"This checkpoint expects task {self.spec['task']!r}")
        obs = decode_observation(payload, self.head_mode)
        resize = Resize((self.data_config.img_size, self.data_config.img_size))
        for name, value in list(obs.items()):
            if name.startswith('observation.images.'):
                obs[name] = resize(self.torch.from_numpy(value).permute(2, 0, 1).float())
            elif name.startswith('observation.state'):
                obs[name] = self.torch.from_numpy(value)
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('Another inference is already running')
        try:
            started = time.monotonic()
            with self.torch.inference_mode():
                batch = self.transform.apply(obs, policy_eval=True)
                actions = self.policy.model.sample_actions(
                    batch['images'].unsqueeze(0).to(self.device, dtype=self.torch.bfloat16),
                    batch['img_masks'].unsqueeze(0).to(self.device),
                    batch['lang_tokens'].unsqueeze(0).to(self.device),
                    batch['lang_masks'].unsqueeze(0).to(self.device),
                    batch['state'].unsqueeze(0).to(self.device, dtype=self.torch.bfloat16),
                    image_grid_thw=batch['image_grid_thw'].to(self.device, dtype=self.torch.long),
                )
                batch['actions'] = actions[0].float().cpu()
                output = self.transform.unapply(batch)['action'].numpy()
            raw = self.validate_actions(output[:, self.action_order])
            output, fixed = restore_constant_actions(raw, self.minimum, self.maximum)
            return {'actions': output.tolist(), 'raw_model_actions': raw.tolist(),
                    'constant_action_dimensions': fixed,
                    'constant_values_restored': int(np.count_nonzero(output != raw)),
                    'inference_s': time.monotonic() - started}
        finally:
            self.lock.release()
