"""Complete-line pairing: never crop an image while retaining its full transcription."""
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .common import read_jsonl, LINE_HEIGHT, MAX_LINE_WIDTH


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    style_words: tuple[int, int]
    target_words: tuple[int, int]
    max_width: int
    style_dropout: float = 0.0
    teacher_noise: float = 0.0


def curriculum(kind='eruku', short_steps=None, long_steps=None, max_width=MAX_LINE_WIDTH, *, medium_steps=None):
    if max_width < 768 or max_width % 8:
        raise ValueError('max-width must be a multiple of 8 and at least 768')
    if kind == 'eruku':
        stages = [Stage('short', 65000, (3, 8), (2, 3), 768),
                  Stage('long', 5000, (3, 8), (1, 32), max_width, 0.1)]
    elif kind == 'three-stage':
        stages = [Stage('short', 1000, (3, 8), (1, 3), max_width),
                  Stage('medium', 2000, (3, 8), (1, 8), max_width),
                  Stage('long', 7000, (3, 8), (1, 32), max_width)]
    elif kind == 'emuru':
        # Apply Emuru's combined line word ranges to Eruku's paired representation.
        stages = [Stage('short', 70000, (3, 6), (1, 3), 768, teacher_noise=0.1),
                  Stage('long', 250, (3, 31), (1, 28), max_width)]
    else:
        raise ValueError(f'Unknown curriculum: {kind}')
    if medium_steps is not None and kind != 'three-stage':
        raise ValueError('medium-steps requires the three-stage curriculum')
    steps = [short_steps, medium_steps, long_steps] if kind == 'three-stage' else [short_steps, long_steps]
    from dataclasses import replace
    stages = [replace(s, steps=n if n is not None else s.steps)
              for s, n in zip(stages, steps, strict=True)]
    if any(s.steps < 0 for s in stages):
        raise ValueError('stage steps cannot be negative')
    return stages


class LinePairs(Dataset):
    def __init__(self, manifest, split, stage, kind='eruku', pair_by='page_id', use='normalized', height=LINE_HEIGHT):
        self.stage, self.kind, self.use, self.height = stage, kind, use, height
        rows = [r for r in read_jsonl(Path(manifest)) if r['split'] == split]
        groups = defaultdict(list)
        for row in rows:
            source_height, source_width = int(row['height']), int(row['width'])
            if source_height <= 0:
                raise ValueError(f"{row['key']}: invalid manifest image height")
            if source_width <= 0:
                continue
            # Filter using training geometry, while retaining original dimensions
            # to detect stale/corrupted manifests when the file is opened.
            width = max(1, round(source_width * height / source_height))
            if width <= stage.max_width:
                row = dict(row, width=width, height=height,
                           _source_width=source_width, _source_height=source_height)
                groups[str(row[pair_by])].append(row)
        self.pairs = []
        for members in groups.values():
            styles = [r for r in members if max(3, stage.style_words[0]) <= len(r['text'].split()) <= stage.style_words[1]]
            for target in members:
                if not stage.target_words[0] <= len(target['text'].split()) <= stage.target_words[1]:
                    continue
                candidates = []
                for style in styles:
                    words = len(style['text'].split()) + len(target['text'].split())
                    if style['key'] == target['key'] or (style.get('xml_path') == target.get('xml_path') and
                        style.get('source_line_id', style['key']) == target.get('source_line_id', target['key'])):
                        continue
                    if kind == 'emuru' and not ((4 <= words <= 7) if stage.name == 'short' else (1 <= words <= 32)):
                        continue
                    if kind == 'emuru' and style['width'] + target['width'] > stage.max_width:
                        continue
                    candidates.append(style)
                if candidates:
                    self.pairs.append((target, candidates))
        if not self.pairs:
            raise ValueError(f'No {split} pairs for stage {stage.name}. Check word counts, widths and grouping; '
                             f'style lines need {max(3, stage.style_words[0])}–{stage.style_words[1]} words and targets '
                             f'need {stage.target_words[0]}–{stage.target_words[1]} words from distinct lines in the same group. '
                             'Use PAGE Word coordinates to extract aligned spans, or set --short-steps 0 for long-stage adaptation.')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        target, candidates = self.pairs[index]
        style = random.choice(candidates)
        def load(row):
            with Image.open(row[f'{self.use}_path']) as im:
                if im.height != row['_source_height'] or im.width != row['_source_width']:
                    raise ValueError(f"Image geometry differs from manifest: {row['key']}")
                image = im.convert('RGB')
                if image.size != (row['width'], self.height):
                    image = image.resize((row['width'], self.height), Image.Resampling.LANCZOS)
                return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        return {'style_img': load(style), 'gen_img': load(target),
                'style_text': style['text'], 'gen_text': target['text']}


def collate_pairs(samples):
    def images(key):
        widths = [s[key].shape[-1] for s in samples]
        width = (max(widths) + 7) // 8 * 8
        # Padding, rather than clipping or squeezing, preserves aligned text.
        return torch.stack([torch.nn.functional.pad(s[key], (0, width-s[key].shape[-1]), value=1)
                            for s in samples]), torch.tensor([(w + 7)//8*8 for w in widths])
    style, sw = images('style_img')
    target, tw = images('gen_img')
    return {'style_img': style, 'gen_img': target, 'style_img_width': sw, 'gen_img_width': tw,
            'style_text': [s['style_text'] for s in samples], 'gen_text': [s['gen_text'] for s in samples],
            'is_historical': torch.tensor([s.get('is_historical', True) for s in samples], dtype=torch.bool)}
