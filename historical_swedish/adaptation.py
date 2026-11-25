"""Height-aware initialization and explicit adapter warm-up configuration."""
import torch
from torch.nn import functional as F

ADAPTERS = ('query_emb', 't5_to_vae', 'diffusion_head')


def resize_adapter(name, source, target, channels):
    """Interpolate each latent channel vertically; preserve input projection scale.

    Flattened latents use (channel, height) order. Never interpolate across channels.
    """
    if name not in {'query_emb.weight', 't5_to_vae.weight', 't5_to_vae.bias'}:
        raise ValueError(f'Unsupported adapter: {name}')
    if name == 'query_emb.weight':
        old, new = source.shape[1], target.shape[1]
        if source.shape[0] != target.shape[0]:
            raise ValueError('Adapter hidden dimensions differ')
        values = source.reshape(source.shape[0], channels, old // channels)
    else:
        old, new = source.shape[0], target.shape[0]
        if source.shape[1:] != target.shape[1:]:
            raise ValueError('Adapter hidden dimensions differ')
        values = source.reshape(channels, old // channels, -1).permute(0, 2, 1)
    if old % channels or new % channels:
        raise ValueError('Adapter dimensions must be divisible by latent channels')
    resized = F.interpolate(values.float(), size=new // channels, mode='linear', align_corners=False)
    if name == 'query_emb.weight':
        resized = resized.reshape_as(target) * (old / new)
    else:
        resized = resized.permute(0, 2, 1).reshape_as(target)
    return resized.to(dtype=target.dtype)


def configure_training(model, adapter_only=False, freeze_decoder=False, freeze_decoder_ffn=False):
    # Set before wrapping with DDP: frozen parameters must stay frozen for the run.
    for name, parameter in model.named_parameters():
        frozen_component = name.startswith(('vae.', 'ocr.')) or (
            getattr(model, 'latent_head', 'linear') == 'diffusion' and name.startswith('t5_to_vae.'))
        adapter = name.startswith(tuple(prefix + '.' for prefix in ADAPTERS))
        decoder = name.startswith('T5.decoder.')
        decoder_ffn = name.startswith('T5.decoder.block.') and '.layer.2.' in name
        parameter.requires_grad_(not frozen_component and (adapter or not adapter_only)
                                 and not (freeze_decoder and decoder)
                                 and not (freeze_decoder_ffn and decoder_ffn))
    # The shared token embedding is canonicalized as T5.shared, so remains
    # trainable for text encoding even though the decoder exposes an alias.
    model.freeze_decoder = freeze_decoder
    model.freeze_decoder_ffn = freeze_decoder_ffn
    model.train(model.training)


def optimizer_groups(model, lr, adapter_lr=None):
    if adapter_lr is None:
        return [dict(params=[p for p in model.parameters() if p.requires_grad], lr=lr, name='all')]
    adapter, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (adapter if name.startswith(tuple(p + '.' for p in ADAPTERS)) else other).append(parameter)
    return [dict(params=params, lr=rate, name=name) for params, rate, name in
            ((adapter, adapter_lr if adapter_lr is not None else lr, 'adapters'),
             (other, lr, 'transformer')) if params]
