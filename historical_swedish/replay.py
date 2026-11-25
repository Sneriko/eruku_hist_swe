"""Streaming local FontSquare paired shards for historical training replay."""
import glob
import json
import math
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


def resolve_shards(pattern):
    """Accept a quoted local glob, one tar, or a newline-separated shard list."""
    path = Path(pattern).expanduser()
    if path.suffix == '.txt' and path.is_file():
        shards = [str((path.parent / line.strip()).resolve())
                  for line in path.read_text().splitlines() if line.strip()]
    else:
        shards = sorted(glob.glob(str(path)))
    if not shards or any(not Path(s).is_file() or not s.endswith('.tar') for s in shards):
        raise ValueError('Replay requires existing local .tar shards (glob or .txt list)')
    return shards


def shard_samples(shards):
    """Read consecutive WebDataset sample members without extracting files."""
    for shard in shards:
        with tarfile.open(shard, 'r|') as archive:
            current, fields = None, {}
            for member in archive:
                if not member.isfile() or '.' not in member.name:
                    continue
                key, suffix = member.name.split('.', 1)
                if key != current:
                    if fields:
                        yield fields
                    current, fields = key, {}
                if suffix in {'json', 'style.rgb.png', 'gen.rgb.png', 'style.bw.png', 'gen.bw.png'}:
                    with archive.extractfile(member) as stream:
                        fields[suffix] = stream.read()
            if fields:
                yield fields


def decode_pair(sample, stage, kind, height, use):
    """Reject unsuitable pairs before resizing; preserve complete transcriptions."""
    import io
    metadata = json.loads(sample['json'])
    style_text, gen_text = metadata['style_text'], metadata['gen_text']
    sw, tw = len(style_text.split()), len(gen_text.split())
    if not (max(3, stage.style_words[0]) <= sw <= stage.style_words[1] and
            stage.target_words[0] <= tw <= stage.target_words[1]):
        return None
    if kind == 'emuru' and not (sw + tw <= (7 if stage.name == 'short' else 32)):
        return None
    suffix = 'rgb.png' if use == 'raw' else 'bw.png'
    images = []
    widths = []
    for key in ('style', 'gen'):
        with Image.open(io.BytesIO(sample[f'{key}.{suffix}'])) as image:
            width = max(1, round(image.width * height / image.height))
            widths.append(width)
            if width > stage.max_width:
                return None
            images.append(image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS))
    if kind == 'emuru' and sum(widths) > stage.max_width:
        return None
    return dict(style_text=style_text, gen_text=gen_text,
                style_img=torch.from_numpy(np.array(images[0], copy=True)).permute(2, 0, 1).float()/255,
                gen_img=torch.from_numpy(np.array(images[1], copy=True)).permute(2, 0, 1).float()/255)


REPLAY_SAMPLING_VERSION = 2


class ReplayMix(IterableDataset):
    """Each mixed epoch visits every historical target exactly once.

    Accelerate shards the resulting stream across ranks. Each worker reads its
    own replay stream, so even a single supplied shard works with multiple workers.
    """
    def __init__(self, historical, shards, historical_probability=0.9, seed=24):
        if not 0 < historical_probability <= 1 or len(historical) == 0:
            raise ValueError('Replay mixing needs historical data and a probability in (0, 1]')
        self.historical, self.shards = historical, shards
        self.historical_probability, self.seed = historical_probability, seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.historical) / self.historical_probability)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def replay(self, rng):
        while True:
            shards = list(self.shards)
            rng.shuffle(shards)
            found = False
            for sample in shard_samples(shards):
                pair = decode_pair(sample, self.historical.stage, self.historical.kind,
                                   self.historical.height, self.historical.use)
                if pair is not None:
                    found = True
                    yield pair
            if not found:
                raise ValueError(f'No eligible synthetic replay pairs for stage {self.historical.stage.name}; '
                                 'check style/target word counts and resized widths, or supply more shards')

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        epoch_seed = self.seed + self.epoch * 1000003
        # Every worker builds the SAME global plan, then takes disjoint positions.
        # Partitioning independently shuffled historical pools would duplicate
        # some targets and omit others when num_workers > 1.
        rng = random.Random(epoch_seed)
        targets = list(range(len(self.historical)))
        rng.shuffle(targets)
        sources = [True] * len(targets) + [False] * (len(self) - len(targets))
        rng.shuffle(sources)
        replay = self.replay(random.Random(epoch_seed + 7919 * (worker_id + 1)))
        historical_position = 0
        try:
            for position, historical in enumerate(sources):
                if historical:
                    index = targets[historical_position]
                    historical_position += 1
                if position % workers != worker_id:
                    continue
                if historical:
                    yield dict(self.historical[index], is_historical=True)
                else:
                    yield dict(next(replay), is_historical=False)
        finally:
            replay.close()
