"""Load final model files or Accelerate checkpoint directories with architecture metadata."""
import json
from pathlib import Path
import torch

HEAD_DEFAULTS = dict(latent_head='linear', diffusion_width=256, diffusion_depth=3,
                     diffusion_steps=20, diffusion_latent_scale=1.0)


def head_options(metadata):
    options = metadata.get('head_config', metadata.get('args', metadata))
    return {key: options.get(key, default) for key, default in HEAD_DEFAULTS.items()}


def load_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        progress = json.loads((path / 'progress.json').read_text())
        args = progress['args']
        metadata = dict(height=progress['height'], max_width=args['max_width'],
                        t5_config=args['t5_config'], vae=args['vae'], head_config=head_options(progress))
        if (path / 'model.safetensors').exists():
            from safetensors.torch import load_file
            state = load_file(str(path / 'model.safetensors'))
        else:
            state = torch.load(path / 'pytorch_model.bin', map_location='cpu', weights_only=False)
    else:
        metadata = torch.load(path, map_location='cpu', weights_only=False)
        state = metadata.get('model', metadata)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    # safetensors omits shared T5 aliases. Restore them for strict loading.
    for key in ('T5.encoder.embed_tokens.weight', 'T5.decoder.embed_tokens.weight'):
        if key not in state and 'T5.shared.weight' in state:
            state[key] = state['T5.shared.weight']
    return metadata, state
