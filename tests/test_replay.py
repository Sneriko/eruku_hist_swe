import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from torch.utils.data import DataLoader
from historical_swedish.data import curriculum, collate_pairs
from historical_swedish.replay import decode_pair, resolve_shards, ReplayMix


def sample(width=40, style='fyra ord i stilen', target='ſvensk åäö'):
    image = io.BytesIO()
    Image.new('RGB', (width, 64), 'white').save(image, format='PNG')
    return {'json': json.dumps(dict(style_text=style, gen_text=target)).encode(),
            **{f'{key}.{kind}.png': image.getvalue() for key in ('style', 'gen') for kind in ('rgb', 'bw')}}


def write_shard(path, samples):
    with tarfile.open(path, 'w') as tar:
        for i, fields in enumerate(samples):
            for key, value in fields.items():
                info = tarfile.TarInfo(f'{i:06d}.{key}')
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))


class ReplayTests(unittest.TestCase):
    def test_resize_filter_and_unicode(self):
        stage = curriculum()[0]
        self.assertIsNotNone(decode_pair(sample(style='three words only'), stage, 'eruku', 192, 'raw'))
        pair = decode_pair(sample(), stage, 'eruku', 192, 'normalized')
        self.assertEqual(tuple(pair['style_img'].shape), (3, 192, 120))
        self.assertEqual(pair['gen_text'], 'ſvensk åäö')
        for fields in (sample(width=800), sample(style='two words'), sample(target='one')):
            self.assertIsNone(decode_pair(fields, stage, 'eruku', 192, 'raw'))
        fields = sample(style='fyra ord i stilen', target='ett två tre fyra')
        self.assertIsNone(decode_pair(fields, curriculum('emuru')[0], 'emuru', 192, 'raw'))

    def test_stream_repeat_workers_and_empty(self):
        class Historical:
            stage, kind, height, use = curriculum()[0], 'eruku', 192, 'normalized'
            def __len__(self):
                return 20
            def __getitem__(self, index):
                return decode_pair(sample(target='historisk text'), self.stage, self.kind, self.height, self.use)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_shard(root/'a.tar', [sample(style='too short'), sample()])
            (root/'shards.txt').write_text('a.tar\n')
            shards = resolve_shards(str(root/'*.tar'))
            self.assertEqual(shards, resolve_shards(str(root/'shards.txt')))
            mixed = ReplayMix(Historical(), shards, 0.5)
            records = list(mixed)
            self.assertEqual({s['is_historical'] for s in records}, {True, False})
            first = [s['gen_text'] for s in records]
            self.assertIn('historisk text', first)
            self.assertIn('ſvensk åäö', first)
            self.assertEqual(len(records), 40)
            self.assertEqual(sum(s['is_historical'] for s in records), 20)
            self.assertEqual(first, [s['gen_text'] for s in mixed])
            mixed.set_epoch(1)
            self.assertNotEqual(first, [s['gen_text'] for s in mixed])
            batches = list(DataLoader(mixed, batch_size=2, num_workers=2, collate_fn=collate_pairs))
            self.assertEqual(sum(len(b['gen_text']) for b in batches), 40)
            write_shard(root/'empty.tar', [sample(style='too short')])
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                next(ReplayMix(Historical(), [str(root/'empty.tar')], 0.5).replay(__import__('random').Random(1)))

    def test_historical_coverage_without_replacement_across_workers(self):
        class IndexedHistorical:
            stage, kind, height, use = curriculum()[0], 'eruku', 64, 'raw'
            def __len__(self):
                return 17
            def __getitem__(self, index):
                return decode_pair(sample(target=f'history-{index} text'), self.stage,
                                   self.kind, self.height, self.use)
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory)/'replay.tar'
            write_shard(shard, [sample()])
            for workers in (0, 3):
                mixed = ReplayMix(IndexedHistorical(), [str(shard)], .5, seed=24)
                batches = list(DataLoader(mixed, batch_size=3, num_workers=workers,
                                         collate_fn=collate_pairs))
                texts = [text for batch in batches for text, historic in
                         zip(batch['gen_text'], batch['is_historical']) if historic]
                self.assertEqual(len(texts), 17)
                self.assertEqual(len(set(texts)), 17)
                self.assertEqual(set(texts), {f'history-{i} text' for i in range(17)})
                self.assertEqual(sum(len(b['gen_text']) for b in batches), 34)
                first = [s['gen_text'] for s in mixed if s['is_historical']]
                mixed.set_epoch(1)
                second = [s['gen_text'] for s in mixed if s['is_historical']]
                self.assertEqual(set(first), set(second))
                self.assertNotEqual(first, second)
            pure = ReplayMix(IndexedHistorical(), [], 1)
            self.assertEqual(len(list(pure)), 17)
            with self.assertRaisesRegex(ValueError, 'probability'):
                ReplayMix(IndexedHistorical(), [], 0)

    def test_shuffled_checkpoint_resume_contract(self):
        from unittest.mock import patch
        from historical_swedish.common import write_jsonl
        from historical_swedish.train import main
        from eruku_continuous_inf import DDPCompatibleEmuru
        from test_historical import tiny_model
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=root/'line.png'
            Image.new('RGB',(24,64),'white').save(image)
            rows=[dict(key=f'{split}{i}',split=split,page_id=split,width=24,height=64,
                       text='historical target' if i==0 else 'four words for style',
                       raw_path=str(image),normalized_path=str(image))
                  for split in ('train','val') for i in range(2)]
            write_jsonl(root/'manifest.jsonl',rows)
            shard=root/'replay.tar'
            write_shard(shard,[sample()])
            argv=['train','--manifest',str(root/'manifest.jsonl'),'--output-dir',str(root/'run'),
                  '--short-steps','0','--long-steps','1','--batch-size','1','--accumulation','1',
                  '--precision','no','--use','raw','--replay-train-pattern',str(shard),
                  '--historical-probability','0.5','--save-every','1','--eval-batches','1',
                  '--no-samples','--no-tensorboard']
            def run(args):
                with patch('historical_swedish.train.DDPCompatibleEmuru',side_effect=lambda *a,**kw:tiny_model(64,DDPCompatibleEmuru)),patch('sys.argv',args):
                    main()
            run(argv)
            checkpoint=root/'run/stage-1-step-1'
            progress=json.loads((checkpoint/'progress.json').read_text())
            self.assertEqual(progress['replay_sampling_version'],2)
            run(argv+['--long-steps','2','--resume',str(checkpoint)])
            ratio_change=argv+['--long-steps','2','--resume',str(checkpoint),
                              '--historical-probability','0.8','--output-dir',str(root/'ratio-branch')]
            with self.assertRaisesRegex(ValueError,'replay ratio differs'):
                run(ratio_change)
            run(ratio_change+['--allow-replay-ratio-change'])
            branch=root/'ratio-branch/stage-1-step-2'
            self.assertEqual(json.loads((branch/'progress.json').read_text())['args']['historical_probability'],0.8)
            self.assertEqual(json.loads((checkpoint/'progress.json').read_text())['args']['historical_probability'],0.5)
            import torch
            optimizer=torch.load(branch/'optimizer.bin',map_location='cpu',weights_only=False)
            self.assertEqual({int(state['step']) for state in optimizer['state'].values()},{2})
            other_shard=root/'other.tar'
            write_shard(other_shard,[sample()])
            with self.assertRaisesRegex(ValueError,'replay source differs'):
                run(ratio_change+['--allow-replay-ratio-change','--replay-train-pattern',str(other_shard)])
            changed=argv+['--use','normalized','--resume',str(checkpoint)]
            with self.assertRaisesRegex(ValueError,'image mode differs'):
                run(changed)
            progress['replay_sampling_version']=1
            (checkpoint/'progress.json').write_text(json.dumps(progress))
            with self.assertRaisesRegex(ValueError,'replay sampling differs'):
                run(argv+['--resume',str(checkpoint)])
