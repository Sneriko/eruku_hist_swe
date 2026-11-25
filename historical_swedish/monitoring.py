"""Checkpoint previews with isolated randomness and reproducible style selection."""
from base64 import b64encode
from contextlib import contextmanager, nullcontext
from pathlib import Path
import json
import html
import random

import numpy as np
from PIL import Image
import torch

from .generate import render, render_latents, decode_line
from .diagnostics import edge_diagnostics, image_diagnostics


@contextmanager
def isolated_rng(seed, device=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device is not None and device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def select_styles(directory, count, height, max_width, seed):
    paths = sorted(p for p in directory.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'})
    random.Random(seed).shuffle(paths)
    eligible = []
    for path in paths:
        with Image.open(path) as image:
            if round(image.width * height / image.height) > max_width:
                continue
        transcript = path.with_suffix('.txt')
        text = transcript.read_text(encoding='utf-8').strip() if transcript.exists() else ''
        if transcript.exists() and len(text.split()) < 3:
            continue
        eligible.append((path, text))
        if len(eligible) == count:
            return eligible
    raise ValueError(f'Need {count} eligible style images in {directory}; found {len(eligible)} '
                     '(check resized widths and three-word minimum for .txt transcriptions)')


def select_validation_styles(dataset, count, seed, sample_max_width, fixed_count=None, iteration=0, *, stratify_lengths=False):
    """Reproducible fixed pairs plus disjoint rotating blocks of held-out pairs.

    Rotation is indexed by checkpoint, so selection is reproducible on resume.
    The rotating pool is exhausted before wrapping; fixed targets are excluded.
    """
    fixed_count = count if fixed_count is None else fixed_count
    if not 0 <= fixed_count <= count or iteration < 0:
        raise ValueError('Invalid fixed preview count or iteration')
    rng = random.Random(seed)
    pairs = list(dataset.pairs)
    rng.shuffle(pairs)
    eligible = [(target, candidates) for target, candidates in pairs
                if target['width'] <= sample_max_width]
    if stratify_lengths:
        from itertools import zip_longest
        by_length = [[], [], []]
        for pair in eligible:
            words = len(pair[0]['text'].split())
            length = 0 if words <= 3 else 1 if words <= 8 else 2
            by_length[length].append(pair)
        eligible = [pair for group in zip_longest(*by_length) for pair in group if pair is not None]
    if len(eligible) < count:
        raise ValueError(f'Need {count} validation preview pairs within sample-max-width; found {len(eligible)}')
    fixed = eligible[:fixed_count]
    pool = eligible[fixed_count:]
    rotating_count = count - fixed_count
    rotating = [pool[(iteration * rotating_count + i) % len(pool)] for i in range(rotating_count)]
    chosen = []
    for index, (target, candidates) in enumerate(fixed + rotating):
        # A pair keeps its reference whenever it is selected, independently of
        # other selections or the checkpoint number. Never touch training RNG.
        key = str(target.get('key', target[f'{dataset.use}_path']))
        style = random.Random(f'{seed}:{key}').choice(candidates)
        chosen.append(dict(style_path=Path(style[f'{dataset.use}_path']),
                           style_text=style['text'], text=target['text'],
                           target_path=Path(target[f'{dataset.use}_path']),
                           target_key=key,
                           split=target.get('split', 'val'),
                           preview_group='fixed' if index < fixed_count else 'rotating'))
    return chosen


def checkpoint_samples(model, styles, output, text, max_width, cfg_scale, seed, writer, step, sample_max_width=None, *,
                       teacher_forced=False, alt_vae=None, recognizer=None, independent_pairs=False):
    output.mkdir(parents=True, exist_ok=True)
    training = model.training
    records = []
    tag = 'samples' if output.name == 'samples' else f'samples/{output.name}'
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.inference_mode(), isolated_rng(seed, device):
            for index, entry in enumerate(styles):
                if isinstance(entry, dict):
                    path, style_text, target_text = entry['style_path'], entry['style_text'], entry['text']
                else:
                    path, style_text = entry
                    target_text = text
                with Image.open(path) as source:
                    style = source.convert('RGB')
                reference = style.resize((max(8, round(style.width * model.image_height / style.height)),
                                          model.image_height), Image.Resampling.LANCZOS)
                reference.save(output / f'{index:02d}-style.png')
                record = dict(style_path=str(path), style_text=style_text, text=target_text)
                if isinstance(entry, dict):
                    record.update({key: entry[key] for key in ('preview_group', 'target_key', 'split') if key in entry})
                if isinstance(entry, dict) and 'target_path' in entry:
                    record['target_path'] = str(entry['target_path'])
                    with Image.open(entry['target_path']) as source:
                        target = source.convert('RGB')
                        target = target.resize((max(8, round(target.width * model.image_height / target.height)),
                                                model.image_height), Image.Resampling.LANCZOS)
                    target.save(output / f'{index:02d}-target.png')
                    if writer:
                        writer.add_image(f'{tag}/{index:02d}/target', np.array(target), step, dataformats='HWC')
                render_args = (model, style, target_text, style_text, sample_max_width or max_width, cfg_scale, max_width)
                if teacher_forced:
                    with isolated_rng(seed, device):
                        gold_latent, _ = render_latents(*render_args, teacher_target=target)
                        gold_image = decode_line(model.vae, gold_latent)
                        gold_image.save(output / f'{index:02d}-teacher-forced.png')
                        record['teacher_forced'] = dict(width=gold_image.width, **edge_diagnostics(gold_image))
                try:
                    with isolated_rng(seed, device) if independent_pairs else nullcontext():
                        if alt_vae is None:
                            generated, stopped = render(*render_args)
                        else:
                            latent, stopped = render_latents(*render_args)
                            generated = decode_line(model.vae, latent)
                            torch.save(latent.cpu(), output / f'{index:02d}-latent.pt')
                            alternate = decode_line(alt_vae, latent)
                            alternate.save(output / f'{index:02d}-alternate.png')
                            record['alternate'] = dict(width=alternate.width, **edge_diagnostics(alternate))
                    generated.save(output / f'{index:02d}-generated.png')
                    record.update(eog_reached=stopped, width=generated.width, **image_diagnostics(generated), **edge_diagnostics(generated))
                    if writer:
                        writer.add_image(f'{tag}/{index:02d}/generated', np.array(generated), step, dataformats='HW')
                except RuntimeError as error:
                    if str(error) != 'Model emitted no image tokens':
                        raise
                    # Early checkpoints can emit EOG immediately; record this failure explicitly.
                    record.update(error=str(error), eog_reached=False)
                if writer:
                    writer.add_image(f'{tag}/{index:02d}/style', np.array(reference), step, dataformats='HWC')
                (output / f'{index:02d}-text.txt').write_text(
                    f'Target: {target_text}\nStyle: {style_text}\n', encoding='utf-8')
                if writer:
                    writer.add_text(f'{tag}/{index:02d}/text',
                                    f'Target: {target_text}\n\nStyle: {style_text}', step)
                records.append(record)
        if recognizer:
            from .evaluate import edit_distance, normalize
            for kind in ('generated', 'teacher-forced', 'alternate'):
                entries = [r if kind == 'generated' else r.get(kind.replace('-', '_')) for r in records]
                paths = [output / f'{i:02d}-{kind}.png' if entry and 'width' in entry else None
                         for i, entry in enumerate(entries)]
                recognizer.predict([path for path in paths if path is not None])
                for record, entry, path in zip(records, entries, paths, strict=True):
                    if entry is None:
                        continue
                    prediction = recognizer.get(path)['text'] if path is not None else ''
                    reference = normalize(record['text'])
                    entry.update(ocr_text=prediction, char_edits=edit_distance(reference, normalize(prediction)),
                                 reference_chars=len(reference))
        (output / 'samples.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        gallery = ['<!doctype html><meta charset="utf-8"><title>Checkpoint samples</title>',
                   '<style>body{font-family:sans-serif}img{max-width:100%}article{border-bottom:1px solid #ccc;padding:16px}</style>']
        for index, record in enumerate(records):
            gallery.append(f'<article><h2>Example {index:02d}</h2><p>Target text: {html.escape(record["text"])}</p>'
                           f'<p>Style text: {html.escape(record["style_text"])}</p>')
            for kind in ('style', 'target', 'generated', 'teacher-forced', 'alternate'):
                details = record if kind == 'generated' else record.get(kind.replace('-', '_'))
                if kind in ('generated', 'teacher-forced', 'alternate') and not (details and 'width' in details):
                    continue
                image_path = output / f'{index:02d}-{kind}.png'
                if image_path.exists():
                    encoded = b64encode(image_path.read_bytes()).decode('ascii')
                    gallery.append(f'<p>{kind.capitalize()}</p><img src="data:image/png;base64,{encoded}">')
                    if details and 'ocr_text' in details:
                        gallery.append(f'<p>OCR {html.escape(details["ocr_text"])}</p>')
            if 'error' in record:
                gallery.append(f'<p>Generation failed: {html.escape(record["error"])}</p>')
            gallery.append('</article>')
        (output / 'index.html').write_text('\n'.join(gallery), encoding='utf-8')
        def summarize(rows, prefix):
            return {f'{prefix}/success_rate': sum('error' not in r and r['eog_reached'] and not r.get('near_blank', False) for r in rows)/len(rows),
                    f'{prefix}/eog_rate': sum(r['eog_reached'] for r in rows)/len(rows),
                    f'{prefix}/near_blank_rate': sum(r.get('near_blank', False) for r in rows)/len(rows)}
        metrics = summarize(records, tag)
        for key in ('generated', 'teacher_forced', 'alternate'):
            rows = records if key == 'generated' else [r[key] for r in records if key in r]
            valid = [r for r in rows if 'edge_coverage' in r]
            if valid:
                for metric in ('horizontal_edge_flag', 'vertical_edge_flag', 'width'):
                    metrics[f'{tag}/{key}/{metric}'] = sum(r[metric] for r in valid) / len(valid)
            chars = sum(r.get('reference_chars', 0) for r in rows)
            if chars:
                metrics[f'{tag}/{key}/cer'] = sum(r.get('char_edits', 0) for r in rows) / chars
        for group in ('fixed', 'rotating'):
            rows = [r for r in records if r.get('preview_group') == group]
            if rows:
                metrics.update(summarize(rows, f'{tag}/{group}'))
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        if writer:
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)
            writer.flush()
        return metrics
    finally:
        model.train(training)
