"""Crop reproducible same-page PAGE pairs and generate a browsable handwriting gallery."""
import argparse
import html
import json
from pathlib import Path
import random
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from historical_swedish.common import (crop_polygon, find_page_image, normalize_background,
                                      normalize_text, parse_points, to_line_height)
from historical_swedish.extract_page_lines import local_name


def child(node, name):
    return next((n for n in node if local_name(n.tag) == name), None)


def prepare(root, out, count, seed):
    rng = random.Random(seed)
    records, coverage = [], []
    for group in sorted(p for p in root.iterdir() if p.is_dir()):
        pages, issues = [], []
        for xml in sorted(group.rglob('*.xml')):
            if xml.parent.name.lower() != 'page':
                continue
            try:
                document = ET.parse(xml).getroot()
                page = next(n for n in document.iter() if local_name(n.tag) == 'Page')
                image = find_page_image(xml, page.get('imageFilename'))
                lines = []
                for n in page.iter():
                    if local_name(n.tag) != 'TextLine':
                        continue
                    coords, equiv = child(n, 'Coords'), child(n, 'TextEquiv')
                    unicode = child(equiv, 'Unicode') if equiv is not None else None
                    if coords is None or unicode is None or not unicode.text:
                        continue
                    text = normalize_text(unicode.text)
                    points = parse_points(coords.get('points', ''))
                    if len(points) < 3 or not 1 <= len(text.split()) <= 32:
                        continue
                    xs, ys = zip(*points)
                    width = (max(xs) - min(xs) + 17) * 128 / (max(ys) - min(ys) + 17)
                    if not 32 <= width <= 4096:
                        continue
                    lines.append(dict(line_id=n.get('id'), text=text, points=points))
                preferred = [line for line in lines if 3 <= len(line['text'].split()) <= 8]
                fallback = [line for line in lines if line not in preferred]
                rng.shuffle(preferred)
                rng.shuffle(fallback)
                styles = fallback + preferred  # pop preferred lines first; keep ten distinct styles possible
                if len(lines) >= 2 and styles:
                    pages.append(dict(xml=str(xml), image=str(image), lines=lines, styles=styles))
            except Exception as exc:
                issues.append(dict(xml=str(xml), error=str(exc)))
        rng.shuffle(pages)
        selected = []
        # Round-robin pages, using each style line at most once.
        while len(selected) < count and any(page['styles'] for page in pages):
            for page in pages:
                if len(selected) == count:
                    break
                if not page['styles']:
                    continue
                style = page['styles'].pop()
                targets = [line for line in page['lines'] if line['line_id'] != style['line_id']]
                target = rng.choice(targets)
                index = len(selected)
                directory = out / group.name / f'{index:02d}'
                directory.mkdir(parents=True, exist_ok=True)
                crops = {}
                with Image.open(page['image']) as source:
                    source = source.convert('RGB')
                    for name, line in [('style', style), ('target', target)]:
                        raw = to_line_height(crop_polygon(source, line['points'], 8), 128, 4096)
                        if raw is None:
                            break
                        crops[name] = (raw, normalize_background(raw))
                if len(crops) != 2:
                    continue
                for name, (raw, normalized) in crops.items():
                    raw.save(directory / f'{name}-raw.png')
                    normalized.save(directory / f'{name}.png')
                record = dict(group=group.name, example=index, xml_path=page['xml'],
                              page_image=page['image'], style_line=style, target_line=target,
                              style_path=str(directory / 'style.png'),
                              target_path=str(directory / 'target.png'),
                              style_text=style['text'], text=target['text'],
                              generated_path=str(directory / 'generated.png'), seed=seed)
                (directory / 'text.txt').write_text(
                    f"Target: {target['text']}\nStyle: {style['text']}\n", encoding='utf-8')
                (directory / 'source.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
                selected.append(record)
        records.extend(selected)
        coverage.append(dict(group=group.name, eligible_pages=len(pages), selected=len(selected), issues=issues))
        print(f'{group.name}: {len(selected)}/{count} pairs ({len(pages)} eligible pages)', flush=True)
    (out / 'pairs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    (out / 'coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2))
    return records


def gallery(out, records, results):
    lookup = {(r['group'], r['example']): r for r in results}
    settings = json.loads((out / 'settings.json').read_text())
    checkpoint = Path(settings['checkpoint'])
    head = 'diffusion' if settings.get('diffusion_steps') else 'linear'
    parts = ['<!doctype html><meta charset="utf-8"><title>Test-suite handwriting examples</title>',
             '<style>body{font:16px sans-serif;margin:24px;max-width:1600px}img{display:block;max-width:100%;height:auto;margin:8px 0 18px}article{border-top:1px solid #bbb;padding:16px 0}small{color:#555}nav a{display:inline-block;margin:5px}</style>',
             f'<h1>Test-suite handwriting examples · {head} · {html.escape(checkpoint.name)}</h1>',
             f'<p>Checkpoint: {html.escape(checkpoint.parent.name)}. PAGE polygon crops; height 128; training background normalization. Each target is a different line on the same page as its style reference. CFG {settings["cfg"]}; seed {settings["seed"]}. Predictions synthesize the supplied target transcription.</p>',
             f'<p>Generated: {len(results)} / {len(records)}</p><nav>']
    for group in sorted({r['group'] for r in records}):
        parts.append(f'<a href="#{html.escape(group, quote=True)}">{html.escape(group)}</a>')
    parts.append('</nav>')
    previous = None
    for r in records:
        if r['group'] != previous:
            parts.append(f'<h2 id="{html.escape(r["group"], quote=True)}">{html.escape(r["group"])}</h2>')
            previous = r['group']
        result = lookup.get((r['group'], r['example']))
        base = Path(r['style_path']).parent.relative_to(out).as_posix()
        parts.extend([f'<article><h3>Example {r["example"]:02d}</h3>',
                      f'<small>Page: {html.escape(Path(r["page_image"]).name)} · style {html.escape(r["style_line"]["line_id"])} · target {html.escape(r["target_line"]["line_id"])}</small>',
                      f'<p>Style: {html.escape(r["style_text"])}</p><img loading="lazy" src="{html.escape(base)}/style.png">',
                      f'<p>Target text: {html.escape(r["text"])}</p>',
                      f'<b>Real target</b><img loading="lazy" src="{html.escape(base)}/target.png">'])
        if result and result.get('file'):
            parts.append(f'<b>Generated</b><img loading="lazy" src="{html.escape(base)}/generated.png">')
            parts.append(f'<small>EOG: {result.get("eog_reached")} · near blank: {result.get("near_blank")} · {result["seconds"]} s</small>')
        else:
            parts.append(f'<p>{html.escape(str(result.get("error"))) if result else "Pending"}</p>')
        parts.append('</article>')
    (out / 'index.html').write_text('\n'.join(parts), encoding='utf-8')


def accelerate_diffusion_sampling(model):
    """Replay the unchanged fixed-size DDIM math as a CUDA graph.

    Draw noise outside the graph so seeds and RNG consumption match eager sampling.
    Check the first result against eager sampling with the identical RNG state.
    """
    import torch
    head = model.diffusion_head
    eager = head.sample
    cache = {}

    @torch.inference_mode()
    def sample(context, steps=20, unconditional=None, cfg_scale=1.0):
        key = (tuple(context.shape), context.dtype, steps, unconditional is not None, cfg_scale)
        if key not in cache:
            fixed_context = context.clone()
            fixed_unconditional = unconditional.clone() if unconditional is not None else None
            fixed_noise = torch.zeros(*context.shape[:-1], head.latent_dim, device=context.device)

            def compute():
                x = fixed_noise
                times = torch.linspace(1, 0, steps + 1, device=context.device)
                for current, following in zip(times[:-1], times[1:]):
                    t = current.expand(context.shape[:-1])
                    alpha, sigma = head.coefficients(t)
                    v = head(x, t, fixed_context).float()
                    if fixed_unconditional is not None and cfg_scale != 1:
                        u = head(x, t, fixed_unconditional).float()
                        v = u + cfg_scale * (v - u)
                    clean = alpha * x - sigma * v
                    noise = sigma * x + alpha * v
                    next_alpha, next_sigma = head.coefficients(following.expand_as(t))
                    x = next_alpha * clean + next_sigma * noise
                return (x * head.latent_scale).to(context.dtype)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compute()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = compute()
            torch.cuda.current_stream().wait_stream(stream)
            cache[key] = (fixed_context, fixed_unconditional, fixed_noise, output, graph)
            check = True
        else:
            check = False
        fixed_context, fixed_unconditional, fixed_noise, output, graph = cache[key]
        state = torch.cuda.get_rng_state() if check else None
        fixed_context.copy_(context)
        if unconditional is not None:
            fixed_unconditional.copy_(unconditional)
        if not check and hasattr(head, 'inference_noise'):
            fixed_noise.copy_(head.inference_noise)
            graph.replay()
            return output.clone()
        fixed_noise.copy_(torch.randn_like(fixed_noise))
        graph.replay()
        result = output.clone()
        if check:
            after = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(state)
            expected = eager(context, steps, unconditional, cfg_scale)
            torch.cuda.set_rng_state(after)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
            print('CUDA-graph diffusion verified against eager sampling with identical noise', flush=True)
        if hasattr(head, 'inference_noise'):
            fixed_noise.copy_(head.inference_noise)
            graph.replay()
            result = output.clone()
        return result

    head.sample = sample


def render_batch(model, records, seed, verify=False, cfg_scale=1.375):
    """Left-pad style prefixes with explicit attention masks; preserve per-pair RNG."""
    import numpy as np
    import torch
    from historical_swedish.diagnostics import image_diagnostics
    device = next(model.parameters()).device
    sequences, generators = [], []
    with torch.inference_mode():
        for record in records:
            torch.manual_seed(seed)
            with Image.open(record['style_path']) as source:
                image = np.array(source.convert('RGB'), copy=True)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device) / 255
            tensor = torch.nn.functional.pad(tensor, (0, (-tensor.shape[-1]) % 8), value=1)
            latent = model.query_rearrange(model._img_encode(tensor))
            sequences.append(torch.cat([model.sos.weight[None], model.query_emb(latent), model.sog.weight[None]], 1))
            generator = torch.Generator(device=device)
            generator.set_state(torch.cuda.get_rng_state())
            generators.append(generator)
        batch = len(records)
        width = max(s.shape[1] for s in sequences)
        prefix = sequences[0].new_zeros((batch, width, sequences[0].shape[-1]))
        decoder_mask = torch.zeros(batch, width, device=device, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            prefix[i:i+1, -sequence.shape[1]:] = sequence
            decoder_mask[i, -sequence.shape[1]:] = 1
        encoded = model.tokenizer([f"{r['style_text']}<sog>{r['text']}" for r in records], padding=True, return_tensors='pt').to(device)
        text = model.T5.shared(encoded['input_ids'])
        mask = encoded['attention_mask']
        enc = model.T5.encoder(inputs_embeds=text, attention_mask=mask, return_dict=True)
        unc = model.T5.encoder(inputs_embeds=model.uncond_embedding.weight[None].expand_as(text), attention_mask=mask, return_dict=True)
        past = upast = None
        current = prefix
        finished, stopped = [False] * batch, [False] * batch
        tokens, sog = [[] for _ in records], [0] * batch
        for step in range(4096 // 8 + 32):
            result = model.T5(encoder_outputs=enc, attention_mask=mask, decoder_inputs_embeds=current,
                              decoder_attention_mask=decoder_mask, past_key_values=past, use_cache=True, return_dict=True)
            past = result.past_key_values
            conditional = result.logits[:, -1:]
            result = model.T5(encoder_outputs=unc, attention_mask=mask, decoder_inputs_embeds=current,
                              decoder_attention_mask=decoder_mask, past_key_values=upast, use_cache=True, return_dict=True)
            upast = result.past_key_values
            unconditional = result.logits[:, -1:]
            guided = unconditional + cfg_scale * (conditional - unconditional)
            if verify and step < 3:
                # Compare the first row, including its left padding and cached continuation,
                # against ordinary unpadded full-prefix decoding.
                n = int(mask[0].sum())
                reference = model.T5(inputs_embeds=text[:1, :n], attention_mask=mask[:1, :n],
                                     decoder_inputs_embeds=sequences[0], use_cache=False).logits[:, -1:]
                torch.testing.assert_close(conditional[:1], reference, atol=3e-3, rtol=3e-3)
            specials = model.t5_to_special(guided).argmax(-1).flatten().tolist()
            active = []
            for i, special in enumerate(specials):
                if finished[i]:
                    continue
                if special == 1:
                    stopped[i] = finished[i] = True
                elif special == 0:
                    sog[i] += 1
                elif len(tokens[i]) == 512:
                    finished[i] = True
                else:
                    active.append(i)
            if all(finished):
                break
            current = model.eog.weight[None].expand(batch, 1, -1).clone()
            if active:
                if model.latent_head == 'diffusion':
                    noise = torch.zeros(batch, 1, model.diffusion_head.latent_dim, device=device)
                    for i in active:
                        noise[i:i+1] = torch.randn(1, 1, model.diffusion_head.latent_dim, device=device, generator=generators[i])
                    model.diffusion_head.inference_noise = noise
                latent = model.predict_next_latent(conditional, unconditional, cfg_scale)
                if model.latent_head == 'diffusion':
                    del model.diffusion_head.inference_noise
                for i in active:
                    value = latent[i:i+1].clone()
                    tokens[i].append(value)
                    current[i:i+1] = model.query_emb(value)
            for i, special in enumerate(specials):
                if not finished[i] and special == 0:
                    current[i:i+1] = model.sog.weight[None]
            if verify and step < 2:
                sequences[0] = torch.cat([sequences[0], current[:1]], 1)
            decoder_mask = torch.cat([decoder_mask, torch.ones(batch, 1, device=device, dtype=torch.long)], 1)
        # Release the autoregressive caches before decoding large image strips.
        del past, upast, enc, unc, result, conditional, unconditional, guided, current
        torch.cuda.empty_cache()
        outputs = []
        for i, values in enumerate(tokens):
            details = dict(eog_reached=stopped[i], image_tokens=len(values), sog_tokens=sog[i], hit_width_limit=len(values) == 512)
            if not values:
                outputs.append((None, dict(details, error='Model emitted no image tokens')))
                continue
            decoded = model.vae.decode(model.z_rearrange(torch.cat(values, 1))).sample.clamp(-1, 1)
            pixels = ((decoded[0, 0].float().cpu().numpy() + 1) * 127.5).round().astype('uint8')
            image = Image.fromarray(pixels)
            outputs.append((image, dict(details, width=image.width, **image_diagnostics(image))))
        if verify:
            print('Batched left-padding and KV cache verified against unpadded decoding', flush=True)
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_args = json.loads((args.checkpoint / 'progress.json').read_text())['args']
    settings = dict(root=str(args.root.resolve()), checkpoint=str(args.checkpoint.resolve()),
                    count=args.count, seed=args.seed, height=128, cfg=1.375,
                    diffusion_steps=20 if checkpoint_args.get('latent_head', 'linear') == 'diffusion' else None,
                    normalization='historical_swedish.common.normalize_background after resizing', precision='float32')
    settings_path = out / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing output settings differ')
    settings_path.write_text(json.dumps(settings, indent=2))
    pairs = out / 'pairs.json'
    records = json.loads(pairs.read_text()) if pairs.exists() else prepare(args.root.resolve(), out, args.count, args.seed)
    results_path = out / 'results.jsonl'
    results = [json.loads(s) for s in results_path.read_text().splitlines()] if results_path.exists() else []
    gallery(out, records, results)
    if args.prepare_only:
        return
    import torch
    from historical_swedish.checkpoints import load_checkpoint, head_options
    from eruku_continuous_inf import DDPCompatibleEmuru
    metadata, state = load_checkpoint(args.checkpoint)
    model = DDPCompatibleEmuru(metadata['t5_config'], metadata['vae'], ocr_checkpoint=None,
                              image_height=metadata['height'], **head_options(metadata))
    model.load_state_dict(state)
    del state
    model.to('cuda').eval()
    if model.latent_head == 'diffusion':
        accelerate_diffusion_sampling(model)
    done = {(r['group'], r['example']) for r in results}
    pending = [r for r in records if (r['group'], r['example']) not in done]
    if args.limit:
        pending = pending[:args.limit]
    with results_path.open('a', buffering=1) as handle:
        for start in range(0, len(pending), 32):
            started = time.monotonic()
            group = pending[start:start+32]
            outputs = render_batch(model, group, args.seed, verify=(start == 0))
            for record, (image, details) in zip(group, outputs):
                if image is not None:
                    image.save(record['generated_path'])
                result = dict(group=record['group'], example=record['example'],
                              file=record['generated_path'] if image is not None else None,
                              seconds=round((time.monotonic() - started) / len(group), 2), **details)
                handle.write(json.dumps(result) + '\n')
                results.append(result)
                print(f'{len(results)}/{len(records)} {record["group"]}/{record["example"]:02}: {details} ({result["seconds"]}s amortized)', flush=True)
            gallery(out, records, results)
    if len(results) == len(records):
        (out / 'COMPLETE.json').write_text(json.dumps(dict(completed=len(results), expected=len(records))))


if __name__ == '__main__':
    main()
