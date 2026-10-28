"""Load the saved MolmoAct2 LoRA topology without discarding checkpoint keys."""
from pathlib import Path
import inspect


def load_molmoact2(cls, config, model_path):
    from peft.tuners.lora import Linear
    from safetensors import safe_open
    from safetensors.torch import load_model

    weights = Path(model_path) / 'model.safetensors'
    with safe_open(weights, framework='pt', device='cpu') as f:
        keys = set(f.keys())
    policy = cls(config)
    prefix = 'model.base_model.model.lm_head.'
    if prefix + 'lora_A.default.weight' in keys:
        head = policy.model.base_model.model.lm_head
        if not isinstance(head, Linear):
            options = {'config': policy.model.peft_config['default']} if 'config' in inspect.signature(Linear).parameters else dict(
                lora_dropout=config.lora_dropout, init_lora_weights=False)
            policy.model.base_model.model.lm_head = Linear(
                head, adapter_name='default', r=config.lora_rank,
                lora_alpha=config.lora_alpha, **options,
            )
    load_model(policy, str(weights), strict=True, device='cpu')
    return policy.to(config.device).eval()
