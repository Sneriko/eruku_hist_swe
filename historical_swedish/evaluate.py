"""Evaluate saved checkpoint previews; never loads optimizer or generator weights.

Default: inspect logs/metadata only. --ocr adds independent Swedish TrOCR CER.
--visual adds the authors' HWD/FID/BFID/KID implementations (optional hwd package).
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
import warnings
from pathlib import Path
import unicodedata


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def normalize(text):
    # Preserve case, punctuation, historical spelling, and internal whitespace.
    return unicodedata.normalize('NFC', text)


def error_rates(rows, prediction):
    chars = sum(len(normalize(r['text'])) for r in rows)
    words = sum(len(normalize(r['text']).split()) for r in rows)
    if not chars or not words:
        raise ValueError('Evaluation requires nonempty reference transcriptions')
    return dict(cer=sum(edit_distance(normalize(r['text']), normalize(r[prediction])) for r in rows)/chars,
                wer=sum(edit_distance(normalize(r['text']).split(), normalize(r[prediction]).split()) for r in rows)/words)


def validation_logs(root):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    path = root / 'tensorboard'
    if not path.exists():
        return {}
    events = EventAccumulator(str(path), size_guidance={'scalars': 0})
    events.Reload()
    out = {}
    for tag in events.Tags()['scalars']:
        if tag.startswith('validation/'):
            for event in events.Scalars(tag):
                out.setdefault(event.step, {})[tag] = event.value
    return out


def exposure(progress, world_size):
    from .data import LinePairs, Stage
    a = progress['args']
    stage = Stage(**progress['stages'][progress['stage']])
    n = len(LinePairs(a['manifest'], 'train', stage, a['curriculum'], a['pair_by'], a['use'], progress['height']))
    mixed = math.ceil(n/a['historical_probability']) if a.get('replay_train_pattern') else n
    # Accelerate flushes accumulation at the end of each data traversal.
    batches = math.ceil(math.ceil(mixed/a['batch_size'])/world_size)
    updates = math.ceil(batches/a['accumulation'])
    complete, remainder = divmod(progress['step'], updates)
    fraction = min(1., remainder*a['accumulation']/batches)
    return dict(eligible_historical_targets=n, mixed_pairs_per_pass=mixed,
                optimizer_updates_per_pass=updates, estimated_passes=complete+fraction,
                data_epoch_index=progress.get('data_epoch'), world_size_assumed=world_size,
                nominal_pairs_per_update=a['batch_size']*a['accumulation']*world_size,
                exposure_caveat='Stage-local estimate; assumes unchanged loader, no mid-pass resumes. Worker/rank tail batches can affect rounding; data_epoch is the recorded traversal index.')


class Recognizer:
    def __init__(self, source, device, batch_size, max_tokens, cache_path):
        import torch
        from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
        from huggingface_hub import hf_hub_download
        self.torch, self.device = torch, device
        self.batch_size, self.max_tokens = batch_size, max_tokens
        local = Path(source)
        config_path = local/'preprocessor_config.json' if local.is_dir() else Path(hf_hub_download(source, 'preprocessor_config.json'))
        config = json.loads(config_path.read_text())
        self.processor = ViTImageProcessor(**config.get('image_processor', config))
        self.tokenizer = AutoTokenizer.from_pretrained(source, extra_special_tokens={})
        self.model = VisionEncoderDecoderModel.from_pretrained(source).to(device).eval()
        self.cache_path = cache_path
        # Model/config/tokenizer identity invalidates predictions when revised.
        identity = json.dumps({'source': source, 'config': self.model.config.to_dict(),
            'generation': self.model.generation_config.to_dict(), 'processor': self.processor.to_dict(),
            'vocab': self.tokenizer.get_vocab(), 'decode_spaces_between_special_tokens': False, 'max_tokens': max_tokens}, sort_keys=True)
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def key(self, path):
        return hashlib.sha256(self.identity.encode()+path.read_bytes()).hexdigest()

    def predict(self, paths):
        from PIL import Image
        todo = {}
        for path in paths:
            key = self.key(path)
            if key not in self.cache:
                todo[key] = path
        todo = list(todo.items())
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start:start+self.batch_size]
            images = []
            for _, path in batch:
                with Image.open(path) as im:
                    images.append(im.convert('RGB'))
            pixels = self.processor(images=images, return_tensors='pt').pixel_values.to(self.device)
            with self.torch.inference_mode():
                kwargs = {'interpolate_pos_encoding': True} if 'interpolate_pos_encoding' in inspect.signature(self.model.get_encoder().forward).parameters else {}
                encoded = self.model.get_encoder()(pixel_values=pixels, return_dict=True, **kwargs)
                ids = self.model.generate(encoder_outputs=encoded, max_new_tokens=self.max_tokens, num_beams=1, use_cache=True)
            texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            for (key, _), tokens, text in zip(batch, ids.tolist(), texts):
                eos = self.model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else eos or []
                # Exclude the decoder-start token from the stop test.
                self.cache[key] = dict(text=text, token_ids=tokens, truncated=not any(t in eos for t in tokens[1:]))
            self.save()

    def save(self):
        tmp = self.cache_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False))
        tmp.replace(self.cache_path)

    def get(self, path):
        return self.cache[self.key(path)]


def configure_hwd_scipy():
    """Adapt SciPy >=1.18's removed disp argument without changing FID math."""
    from scipy import linalg
    from types import SimpleNamespace
    from hwd.metrics.fid import fid_score_crop64x64
    if 'disp' not in inspect.signature(linalg.sqrtm).parameters:
        def sqrtm(matrix, disp=True):
            result = linalg.sqrtm(matrix)
            return result if disp else (result, None)
        fid_score_crop64x64.linalg = SimpleNamespace(sqrtm=sqrtm)


def visual_metrics(rows, names, device):
    import torch
    from PIL import Image
    from hwd.datasets.base_dataset import BaseDataset
    from hwd import scores
    configure_hwd_scipy()
    # Explicit order preserves pairing, and page IDs are style proxies, not known writers.
    authors = [r['style_group'] for r in rows]
    def dataset(key):
        images = []
        for r in rows:
            with Image.open(r[key]) as im:
                images.append(im.convert('RGB'))
        return BaseDataset(Path('.'), images, authors, sorted(set(authors)))
    result = {}
    for name in names:
        torch.manual_seed(24)
        score = getattr(scores, name+'Score')(height=32)
        score.device = device
        score.to(device)
        value = score(dataset('generated_path'), dataset('real_path'))
        result[name.lower()] = float(value)
        del score
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path, help='Checkpoint directory or training run directory')
    p.add_argument('--output', type=Path)
    p.add_argument('--group', choices=['fixed', 'rotating', 'all'], default='fixed')
    p.add_argument('--steps', type=int, nargs='+', help='Evaluate selected stage-local steps')
    p.add_argument('--ocr', action='store_true')
    p.add_argument('--ocr-model', default='Riksarkivet/trocr-large-handwritten-hist-swe-3-char')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--visual', nargs='+', choices=['HWD','FID','BFID','KID'])
    p.add_argument('--world-size', type=int, default=1, help='Number of training processes, NOT evaluation GPUs')
    args = p.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.world_size) <= 0:
        p.error('batch-size, max-new-tokens and world-size must be positive')
    root = args.path.parent if (args.path/'progress.json').exists() else args.path
    checkpoints = [args.path] if (args.path/'progress.json').exists() else list(root.glob('stage-*-step-*'))
    checkpoints = sorted(checkpoints, key=lambda x: tuple(map(int, (x.name.split('-')[1], x.name.split('-')[3]))))
    output = args.output or root/'analysis'
    output.mkdir(parents=True, exist_ok=True)
    logs = validation_logs(root)
    ocr = Recognizer(args.ocr_model, args.device, args.batch_size, args.max_new_tokens, output/'ocr-cache.json') if args.ocr else None
    (output/'settings.json').write_text(json.dumps({
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'normalization': 'Unicode NFC only; case, punctuation and whitespace preserved',
        'ocr_identity': ocr.identity if ocr else None,
        'ocr_revision': getattr(ocr.model.config, '_commit_hash', None) if ocr else None,
        'visual_height': 32, 'visual_implementation': 'aimagelab/HWD',
        'selection_note': 'Preview screening only. Confirm on a larger held-out set before selecting a final model.'
    }, indent=2))
    summary, details, exposures = [], [], {}
    baseline = None
    style_groups = {}
    if args.visual:
        first = json.loads((checkpoints[0]/'progress.json').read_text())
        with Path(first['args']['manifest']).open() as stream:
            for line in stream:
                record = json.loads(line)
                style_groups[str(record.get('key'))] = str(record.get(first['args']['pair_by'], 'unknown'))
    for checkpoint in checkpoints:
        progress = json.loads((checkpoint/'progress.json').read_text())
        if args.steps and progress['step'] not in args.steps:
            continue
        sample_path = checkpoint/'samples'/'samples.json'
        if not sample_path.exists():
            raise FileNotFoundError(f'Checkpoint has no preview manifest: {sample_path}')
        all_rows = json.loads(sample_path.read_text())
        rows = []
        for index, record in enumerate(all_rows):
            if args.group != 'all' and record.get('preview_group') != args.group:
                continue
            row = dict(record)
            row['generated_path'] = sample_path.parent/f'{index:02d}-generated.png'
            row['real_path'] = sample_path.parent/f'{index:02d}-target.png'
            row['style_group'] = style_groups.get(Path(row['style_path']).stem, 'preview-pool')
            if not row['real_path'].exists():
                raise FileNotFoundError(f'Missing held-out target: {row["real_path"]}')
            row['failed'] = 'error' in row or not row['generated_path'].exists()
            rows.append(row)
        if len(rows) < 100 and args.visual:
            warnings.warn('Very small preview set: FID/BFID/KID and HWD are exploratory, not stable benchmark estimates.')
        if not rows:
            raise ValueError(f'No {args.group} samples in {checkpoint}')
        signature = [(r.get('target_key'), r['text'], r['style_path']) for r in rows]
        if args.group == 'fixed':
            if baseline is not None and signature != baseline:
                raise ValueError('Fixed pairs differ between checkpoints; refusing an unfair comparison')
            baseline = signature
        a = progress['args']
        key = json.dumps([progress['stage'], progress['stages'], a['manifest'], a['batch_size'], a['accumulation'], a.get('historical_probability')], sort_keys=True)
        if key not in exposures:
            exposures[key] = exposure(progress, args.world_size)
        # Exposure depends on step; cheap once eligible count is known.
        exp = dict(exposures[key]); updates = exp['optimizer_updates_per_pass']
        full, rest = divmod(progress['step'], updates)
        batches = math.ceil(math.ceil(exp['mixed_pairs_per_pass']/a['batch_size'])/args.world_size)
        exp.update(estimated_passes=full+min(1., rest*a['accumulation']/batches), data_epoch_index=progress.get('data_epoch'))
        global_step = sum(s['steps'] for s in progress['stages'][:progress['stage']])+progress['step']
        result = dict(checkpoint=checkpoint.name, step=global_step, group=args.group, count=len(rows),
            failed_count=sum(r['failed'] for r in rows),
            eog_rate=sum(bool(r.get('eog_reached')) for r in rows)/len(rows),
            near_blank_rate=sum(bool(r.get('near_blank')) for r in rows)/len(rows), **exp, **logs.get(global_step, {}))
        if ocr:
            ocr.predict([r['real_path'] for r in rows]+[r['generated_path'] for r in rows if not r['failed']])
            for r in rows:
                r['real_ocr'] = ocr.get(r['real_path'])['text']
                r['generated_ocr'] = '' if r['failed'] else ocr.get(r['generated_path'])['text']
                r['ocr_truncated'] = (False if r['failed'] else ocr.get(r['generated_path'])['truncated']) or ocr.get(r['real_path'])['truncated']
            generated = error_rates(rows, 'generated_ocr'); real = error_rates(rows, 'real_ocr')
            result.update(cer_generated=generated['cer'], cer_real=real['cer'],
                delta_cer=abs(generated['cer']-real['cer']), signed_delta_cer=generated['cer']-real['cer'],
                wer_generated=generated['wer'], wer_real=real['wer'], ocr_truncated_count=sum(r['ocr_truncated'] for r in rows))
        if args.visual:
            valid = [r for r in rows if not r['failed']]
            if len(valid) < 2:
                raise ValueError('Visual distribution metrics require at least two generated images')
            result.update(visual_metrics(valid, args.visual, args.device), visual_count=len(valid))
        details.extend(dict(checkpoint=checkpoint.name, **{k:str(v) if isinstance(v,Path) else v for k,v in r.items()}) for r in rows)
        summary.append(result)
        (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (output/'predictions.json').write_text(json.dumps(details, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if not summary:
        p.error('No matching checkpoints')
    keys = list(dict.fromkeys(k for row in summary for k in row))
    with (output/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(summary)
    ranking = {}
    for metric in ['validation/loss', 'cer_generated', 'delta_cer', 'hwd', 'fid', 'bfid', 'kid']:
        eligible = [r for r in summary if metric in r]
        if eligible:
            ranking[metric] = sorted((dict(checkpoint=r['checkpoint'], value=r[metric], estimated_passes=r['estimated_passes']) for r in eligible), key=lambda r:r['value'])[:10]
    (output/'ranking.json').write_text(json.dumps(ranking, indent=2))
    print('Saved analysis to', output)


if __name__ == '__main__':
    main()
