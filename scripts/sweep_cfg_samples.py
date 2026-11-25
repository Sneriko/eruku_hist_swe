"""Run a reproducible CFG/seed sweep on saved checkpoint preview pairs."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from historical_swedish.checkpoints import load_checkpoint, head_options
from eruku_continuous_inf import DDPCompatibleEmuru
from historical_swedish.diagnostics import image_diagnostics


@torch.inference_mode()
def render_cached(model, entry, seed, scale, max_width, verify=False):
    torch.manual_seed(seed)
    with Image.open(entry['style_path']) as source:
        style = source.convert('RGB')
    width = max(8, round(style.width * model.image_height / style.height))
    image = style.resize((width, model.image_height), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()[None].to(model.T5.device) / 255
    tensor = torch.nn.functional.pad(tensor, (0, (-width) % 8), value=1)
    latent = model.query_rearrange(model._img_encode(tensor))
    sequence = torch.cat([model.sos.weight[None], model.query_emb(latent), model.sog.weight[None]], 1)
    encoded = model.tokenizer([f"{entry['style_text']}<sog>{entry['text']}"], return_tensors='pt').to(model.T5.device)
    text = model.T5.shared(encoded['input_ids'])
    mask = encoded['attention_mask']
    enc = model.T5.encoder(inputs_embeds=text, attention_mask=mask, return_dict=True)
    unc = model.T5.encoder(inputs_embeds=model.uncond_embedding.weight[None].expand_as(text), attention_mask=mask, return_dict=True) if scale != 1 else None
    past = upast = None
    tokens, stopped, sog = [], False, 0
    current = sequence
    for step in range(max_width // 8 + 32):
        result = model.T5(encoder_outputs=enc, attention_mask=mask, decoder_inputs_embeds=current,
                          past_key_values=past, use_cache=True, return_dict=True)
        past = result.past_key_values
        conditional = result.logits[:, -1:]
        unconditional = None
        output = conditional
        if unc is not None:
            u = model.T5(encoder_outputs=unc, attention_mask=mask, decoder_inputs_embeds=current,
                         past_key_values=upast, use_cache=True, return_dict=True)
            upast = u.past_key_values
            unconditional = u.logits[:, -1:]
            output = unconditional + scale * (conditional - unconditional)
        if verify and step < 3:
            reference = model.T5(inputs_embeds=text, attention_mask=mask, decoder_inputs_embeds=sequence, use_cache=False).logits[:, -1:]
            if unc is not None:
                ur = model.T5(inputs_embeds=model.uncond_embedding.weight[None].expand_as(text), attention_mask=mask,
                              decoder_inputs_embeds=sequence, use_cache=False).logits[:, -1:]
                reference = ur + scale * (reference - ur)
            torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)
        special = model.t5_to_special(output).argmax(-1).item()
        if special == 1:
            stopped = True
            break
        if special == 0:
            sog += 1
            current = model.sog.weight[None]
        else:
            if len(tokens) == max_width // 8:
                break
            latent = model.predict_next_latent(conditional, unconditional, scale)
            tokens.append(latent)
            current = model.query_emb(latent)
        if verify and step < 2:
            sequence = torch.cat([sequence, current], 1)
    details = dict(eog_reached=stopped, image_tokens=len(tokens), sog_tokens=sog,
                   hit_width_limit=len(tokens) == max_width // 8)
    if not tokens:
        return None, {**details, 'error': 'Model emitted no image tokens'}
    decoded = model.vae.decode(model.z_rearrange(torch.cat(tokens, 1))).sample.clamp(-1, 1)
    pixels = ((decoded[0, 0].float().cpu().numpy() + 1) * 127.5).round().astype('uint8')
    image = Image.fromarray(pixels)
    return image, {**details, 'width': image.width, **image_diagnostics(image)}


def write_gallery(out, entries, scales, seeds, records):
    lookup = {(r['sample'], r['seed'], r['cfg']): r for r in records}
    parts = ['<!doctype html><meta charset="utf-8"><title>CFG och seed – steg 3250</title>',
             '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}td{min-width:220px}article{margin-bottom:40px}small{display:block} .scroll{overflow-x:auto}</style>',
             '<h1>CFG- och seedjämförelse</h1><p>Samma checkpoint, måltext och referens. CFG varierar kolumnvis. Seeds varierar radvis. Seed påverkar VAE-samplingen av stilreferensen och eventuell diffusion; genereringen samplar inte texttoken. Alla körningar använder referenstext. EOG betyder inte att texten är korrekt.</p>',
             f'<p>Klart: {len(records)} / {len(entries)*len(scales)*len(seeds)}</p>']
    for i, entry in enumerate(entries):
        parts += [f'<article id="sample-{i:02}"><h2>Exempel {i:02}</h2><p>Mål: {html.escape(entry["text"])}</p><p>Referenstext: {html.escape(entry["style_text"])}</p>',
                  f'<p>Referens</p><img loading="lazy" src="references/{i:02}-style.png"><p>Målbild</p><img loading="lazy" src="references/{i:02}-target.png">',
                  f'<p>Ursprunglig preview</p><img loading="lazy" src="references/{i:02}-original.png">',
                  '<div class="scroll"><table><tr><th>Seed</th>' + ''.join(f'<th>CFG {s:g}</th>' for s in scales) + '</tr>']
        for seed in seeds:
            parts.append(f'<tr><th>{seed}</th>')
            for scale in scales:
                r = lookup.get((i, seed, scale))
                if r is None:
                    parts.append('<td>Väntar</td>')
                else:
                    picture = f'<img loading="lazy" src="{r["file"]}">' if r.get('file') else html.escape(r.get('error', ''))
                    parts.append(f'<td>{picture}<small>EOG: {r["eog_reached"]}; bredd: {r.get("width", 0)}; maxbredd: {r["hit_width_limit"]}</small></td>')
            parts.append('</tr>')
        parts.append('</table></div></article>')
    temporary = out / 'index.tmp.html'
    temporary.write_text('\n'.join(parts))
    temporary.replace(out / 'index.html')
    total = len(entries) * len(scales) * len(seeds)
    average = sum(r['seconds'] for r in records) / len(records) if records else None
    status = dict(completed=len(records), expected=total, complete=len(records) == total,
                  average_seconds=average, estimated_remaining_hours=(total-len(records))*average/3600 if average else None)
    temporary = out / 'status.tmp.json'
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(out / 'status.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scales', type=float, nargs='+', default=[1, 1.125, 1.25, 1.375])
    p.add_argument('--seeds', type=int, nargs='+', default=[24, 42, 123])
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    checkpoint, out = args.checkpoint.resolve(), args.output.resolve()
    if out == checkpoint or checkpoint in out.parents:
        raise ValueError('Use a separate experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads((checkpoint / 'samples/samples.json').read_text())
    if args.limit:
        entries = entries[:args.limit]
    config = json.loads((checkpoint / 'progress.json').read_text())['args']
    settings = dict(checkpoint=str(checkpoint), scales=args.scales, seeds=args.seeds, sample_count=len(entries),
                    max_width=config['sample_max_width'] or config['max_width'], style_text=True, precision='float32', cached=True)
    settings_path = out / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing experiment settings differ')
    settings_path.write_text(json.dumps(settings, indent=2))
    (out / 'samples.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    (out / 'references').mkdir(exist_ok=True)
    for i in range(len(entries)):
        for kind in ['style', 'target', 'generated']:
            source = checkpoint / f'samples/{i:02}-{kind}.png'
            if source.exists():
                shutil.copyfile(source, out / f'references/{i:02}-{"original" if kind == "generated" else kind}.png')
    results = out / 'results.jsonl'
    records = [json.loads(s) for s in results.read_text().splitlines()] if results.exists() else []
    done = {(r['sample'], r['seed'], r['cfg']) for r in records}
    write_gallery(out, entries, args.scales, args.seeds, records)
    print('Loading checkpoint', flush=True)
    metadata, state = load_checkpoint(checkpoint)
    model = DDPCompatibleEmuru(config['t5_config'], config['vae'], ocr_checkpoint=None, image_height=config['height'], **head_options(metadata))
    model.load_state_dict(state)
    model.to('cuda').eval()
    verified = False
    total = len(entries) * len(args.scales) * len(args.seeds)
    with results.open('a', buffering=1) as handle:
        for i, entry in enumerate(entries):
            for seed in args.seeds:
                for scale in args.scales:
                    if (i, seed, scale) in done:
                        continue
                    started = time.monotonic()
                    result, details = render_cached(model, entry, seed, scale, settings['max_width'], verify=not verified)
                    verified = True
                    name = f'seed-{seed}/cfg-{scale:g}/{i:02}-generated.png'
                    if result is not None:
                        path = out / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        result.save(path)
                    r = dict(sample=i, seed=seed, cfg=scale, file=name if result is not None else None,
                             seconds=round(time.monotonic()-started, 2), **details)
                    handle.write(json.dumps(r) + '\n')
                    records.append(r)
                    write_gallery(out, entries, args.scales, args.seeds, records)
                    print(f'{len(records)}/{total}: sample={i:02} seed={seed} cfg={scale:g} {details} ({r["seconds"]}s)', flush=True)
    keys = sorted({k for r in records for k in r})
    with (out / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    (out / 'COMPLETE.json').write_text(json.dumps(dict(completed=len(records), expected=total)))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
