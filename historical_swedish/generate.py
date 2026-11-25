"""Generate native 64px text, with optional proportional resizing for OCR."""
import argparse
import json
from pathlib import Path
from .diagnostics import image_diagnostics
from .checkpoints import load_checkpoint, head_options
import torch
import numpy as np
from PIL import Image
from eruku_continuous_inf import DDPCompatibleEmuru


def encode_line(model, image, max_width):
    device = next(model.parameters()).device
    width = max(8, round(image.width * model.image_height / image.height))
    if width > max_width:
        raise ValueError('Line exceeds max-width after resizing')
    resized = image.convert('RGB').resize((width, model.image_height), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
    tensor = torch.nn.functional.pad(tensor, (0, (-width) % 8), value=1)
    return model.query_rearrange(model._img_encode(tensor))


def decode_line(vae, latent):
    image = vae.decode(latent).sample.clamp(-1, 1)
    pixels = ((image[0, 0].cpu().float().numpy() + 1) * 127.5).round().astype('uint8')
    return Image.fromarray(pixels)


@torch.inference_mode()
def render_latents(model, style, text, style_text='', max_width=2048, cfg_scale=1.25,
                   style_max_width=None, teacher_target=None):
    if max_width < 8 or max_width % 8:
        raise ValueError('max-width must be a positive multiple of 8')
    if cfg_scale < 0:
        raise ValueError('cfg-scale must be nonnegative')
    device = next(model.parameters()).device
    latents = encode_line(model, style, style_max_width or max_width)
    sequence = torch.cat([model.sos.weight[None], model.query_emb(latents), model.sog.weight[None]], dim=1)
    encoded = model.tokenizer([f'{style_text}<sog>{text}'], return_tensors='pt').to(device)
    text_embeds = model.T5.shared(encoded['input_ids'])
    if teacher_target is not None:
        # Predict each column from preceding real columns, at the reference width.
        gold = encode_line(model, teacher_target, max_width)
        sequence = torch.cat([sequence, model.query_emb(gold[:, :-1])], dim=1)
        conditional = model.T5(inputs_embeds=text_embeds, attention_mask=encoded['attention_mask'],
                               decoder_inputs_embeds=sequence).logits[:, -gold.shape[1]:]
        unconditional = None
        if cfg_scale != 1:
            unconditional = model.T5(inputs_embeds=model.uncond_embedding.weight[None].expand_as(text_embeds),
                attention_mask=encoded['attention_mask'], decoder_inputs_embeds=sequence).logits[:, -gold.shape[1]:]
        return model.z_rearrange(model.predict_next_latent(conditional, unconditional, cfg_scale)), None
    generated = []
    stopped = False
    # Bound delimiter predictions too, so malformed models cannot loop forever.
    for _ in range(max_width // 8 + 32):
        conditional = model.T5(inputs_embeds=text_embeds, attention_mask=encoded['attention_mask'],
                               decoder_inputs_embeds=sequence).logits[:, -1:]
        unconditional = None
        if cfg_scale != 1:
            unconditional = model.T5(inputs_embeds=model.uncond_embedding.weight[None].expand_as(text_embeds),
                attention_mask=encoded['attention_mask'], decoder_inputs_embeds=sequence).logits[:, -1:]
        guided = conditional if unconditional is None else unconditional + cfg_scale * (conditional - unconditional)
        special = model.t5_to_special(guided).argmax(-1).item()
        if special == 1:
            stopped = True
            break
        if special == 0:
            sequence = torch.cat([sequence, model.sog.weight[None]], dim=1)
            continue
        if len(generated) == max_width // 8:
            break
        latent = model.predict_next_latent(conditional, unconditional, cfg_scale)
        generated.append(latent)
        sequence = torch.cat([sequence, model.query_emb(latent)], dim=1)
    if not generated:
        raise RuntimeError('Model emitted no image tokens')
    return model.z_rearrange(torch.cat(generated, dim=1)), stopped


@torch.inference_mode()
def render(model, style, text, style_text='', max_width=2048, cfg_scale=1.25, style_max_width=None):
    latent, stopped = render_latents(model, style, text, style_text, max_width, cfg_scale, style_max_width)
    return decode_line(model.vae, latent), stopped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', help='Final model.pth from historical training')
    source.add_argument('--init-checkpoint', help='Original local .pth or Hugging Face model ID')
    p.add_argument('--height', type=int, default=64, help='Height for --init-checkpoint baseline')
    p.add_argument('--t5-config', default='google-t5/t5-large')
    p.add_argument('--vae', default='blowing-up-groundhogs/emuru_vae')
    p.add_argument('--style-image', required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--style-text', default='')
    p.add_argument('--output', required=True)
    p.add_argument('--max-width', type=int)
    p.add_argument('--diffusion-steps', type=int, help='Override denoising steps saved in a diffusion checkpoint')
    p.add_argument('--cfg-scale', type=float, default=1.25)
    p.add_argument('--output-height', type=int, help='Optional final image height, e.g. 192 for OCR; resizes the native output')
    p.add_argument('--seed', type=int, default=24)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.output_height is not None and args.output_height <= 0:
        p.error('--output-height must be positive')
    torch.manual_seed(args.seed)
    if args.checkpoint:
        ckpt, state = load_checkpoint(args.checkpoint)
        model = DDPCompatibleEmuru(ckpt['t5_config'], ckpt['vae'], ocr_checkpoint=None, image_height=ckpt['height'], **head_options(ckpt))
        model.load_state_dict(state)
        default_width = ckpt['max_width']
    else:
        from .train import initialize
        model = DDPCompatibleEmuru(args.t5_config, args.vae, ocr_checkpoint=None, image_height=args.height)
        initialize(model, args.init_checkpoint)
        default_width = 2048
    if args.diffusion_steps is not None:
        if args.diffusion_steps < 1 or model.latent_head != 'diffusion':
            p.error('--diffusion-steps requires a diffusion checkpoint and a positive value')
        model.diffusion_steps = args.diffusion_steps
    model.to(args.device).eval()
    torch.manual_seed(args.seed)
    with Image.open(args.style_image) as style:
        result, stopped = render(model, style, args.text, args.style_text,
                                 args.max_width or default_width, args.cfg_scale)
    native_height, native_width = result.height, result.width
    if args.output_height is not None and args.output_height != native_height:
        result = result.resize((max(1, round(native_width * args.output_height / native_height)),
                                args.output_height), Image.Resampling.LANCZOS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    report = dict(height=result.height, width=result.width, native_height=native_height,
                  native_width=native_width, eog_reached=stopped,
                  seed=args.seed, **image_diagnostics(result))
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    if report['near_blank']:
        print('Warning: generated image is nearly blank; this is not a successful handwriting example')
    print(f'Saved {result.width}×{result.height}; EOG reached: {stopped}')


if __name__ == '__main__':
    main()
