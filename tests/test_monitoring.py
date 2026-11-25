from base64 import b64encode
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from historical_swedish.monitoring import select_styles, select_validation_styles, checkpoint_samples
from historical_swedish.train import main
from historical_swedish.common import write_jsonl
from eruku_continuous_inf import DDPCompatibleEmuru
from test_historical import tiny_model


class MonitoringTests(unittest.TestCase):
    def test_three_word_reference_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in [('short', 'two words'), ('eligible', 'three style words')]:
                Image.new('RGB', (100, 64), 'white').save(root/f'{name}.png')
                (root/f'{name}.txt').write_text(text)
            selected = select_styles(root, 1, 64, 2048, 24)
            self.assertEqual(selected, [(root/'eligible.png', 'three style words')])

    def test_checkpoint_training_events_and_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            styles = root/'styles'
            styles.mkdir()
            for i in range(10):
                Image.new('RGB', (24, 64), 'white').save(styles/f'{i}.png')
                (styles/f'{i}.txt').write_text('fyra ord i stilen')
            rows = [dict(key=f'{split}{i}', split=split, page_id=split, height=64,
                         width=24, text='åäö text' if i == 0 else 'fyra ord i stilen',
                         normalized_path=str(styles/'0.png'))
                    for split in ('train', 'val', 'test') for i in range(51)]
            write_jsonl(root/'lines.jsonl', rows)
            argv = ['train', '--manifest', str(root/'lines.jsonl'), '--output-dir', str(root/'run'),
                    '--short-steps', '2', '--long-steps', '0', '--batch-size', '1',
                    '--accumulation', '2', '--precision', 'no', '--save-every', '1',
                    '--log-every', '1', '--sample-style-dir', str(styles)]
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), \
                 patch('historical_swedish.monitoring.render', return_value=(Image.new('L', (40, 64)), True)) as renderer, \
                 patch('sys.argv', argv):
                main()
            self.assertEqual(renderer.call_count, 100)
            self.assertIn(renderer.call_args.args[2], ('åäö text', 'fyra ord i stilen'))
            for step in (1, 2):
                folder = root/f'run/stage-0-step-{step}/samples'
                self.assertEqual(len(list(folder.glob('*-generated.png'))), 50)
                self.assertEqual(len(list(folder.glob('*-target.png'))), 50)
                self.assertEqual(len(list(folder.glob('*-text.txt'))), 50)
                self.assertTrue((folder/'index.html').exists())
                records = json.loads((folder/'samples.json').read_text())
                self.assertEqual(len(records), 50)
                self.assertTrue(all(r['split'] == 'test' for r in records))
            events = EventAccumulator(str(root/'run/tensorboard')).Reload()
            self.assertEqual([e.step for e in events.Scalars('train/loss')], [1, 2])
            self.assertEqual(len(events.Scalars('validation/loss')), 2)
            self.assertEqual(len(events.Scalars('validation/eog_accuracy')), 2)
            self.assertEqual(len(events.Scalars('train/boundary_ce_loss')), 2)
            self.assertIn('samples/00/generated', events.Tags()['images'])

    def test_rng_and_mode_restored_for_failed_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (24, 64)).save(root/'style.png')
            (root/'style.txt').write_text('fyra ord i stilen')
            styles = select_styles(root, 1, 64, 6144, 24)
            model = torch.nn.Linear(1, 1).train()
            model.image_height = 64
            random.seed(3)
            np.random.seed(3)
            torch.manual_seed(3)
            state = torch.get_rng_state().clone()
            py_state = random.getstate()
            np_state = np.random.get_state()
            with patch('historical_swedish.monitoring.render', side_effect=RuntimeError('Model emitted no image tokens')):
                metrics = checkpoint_samples(model, styles, root/'samples', 'åäö', 6144, 1.25, 24, None, 1)
            self.assertEqual(metrics['samples/success_rate'], 0)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(py_state, random.getstate())
            self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))

    def test_validation_previews_use_target_text_without_target_as_input(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            style_path, target_path = root/'style.png', root/'target.png'
            Image.new('RGB', (32, 192), 'white').save(style_path)
            Image.new('RGB', (48, 192), 'black').save(target_path)
            style = dict(normalized_path=str(style_path), text='style words on page')
            target = dict(normalized_path=str(target_path), text='actual historical target', width=16)
            data = SimpleNamespace(pairs=[(target, [style])], use='normalized')
            state = random.getstate()
            pairs = select_validation_styles(data, 1, 24, 1024)
            self.assertEqual(pairs, select_validation_styles(data, 1, 24, 1024))
            self.assertEqual(state, random.getstate())
            model = torch.nn.Linear(1, 1).eval()
            model.image_height = 64
            with patch('historical_swedish.monitoring.render',
                       return_value=(Image.new('L', (40, 64), 0), True)) as renderer:
                checkpoint_samples(model, pairs, root/'samples', 'unused modern prompt',
                                   2048, 1.25, 24, None, 1, 1024)
            self.assertEqual(renderer.call_args.args[2], target['text'])
            self.assertEqual(renderer.call_args.args[3], style['text'])
            self.assertEqual(renderer.call_args.args[1].size, (32, 192))
            self.assertEqual(Image.open(root/'samples/00-target.png').size, (16, 64))
            record = json.loads((root/'samples/samples.json').read_text())[0]
            self.assertEqual(record['target_path'], str(target_path))
            self.assertIn(target['text'], (root/'samples/00-text.txt').read_text())
            gallery = (root/'samples/index.html').read_text()
            for kind in ('style', 'target', 'generated'):
                png = (root/f'samples/00-{kind}.png').read_bytes()
                encoded = b64encode(png).decode('ascii')
                self.assertIn(f'src="data:image/png;base64,{encoded}"', gallery)
            with self.assertRaisesRegex(ValueError, 'Need 2 validation'):
                select_validation_styles(data, 2, 24, 1024)

    def test_nonblank_runaway_preview_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'style.png'
            Image.new('RGB', (24, 64), 'white').save(path)
            model = torch.nn.Linear(1, 1).eval()
            model.image_height = 64
            with patch('historical_swedish.monitoring.render',
                       return_value=(Image.new('L', (1024, 64), 0), False)):
                metrics = checkpoint_samples(model, [(path, '')], root/'samples',
                                             'text', 2048, 1.25, 24, None, 1)
            self.assertEqual(metrics['samples/near_blank_rate'], 0)
            self.assertEqual(metrics['samples/eog_rate'], 0)
            self.assertEqual(metrics['samples/success_rate'], 0)

    def test_white_previews_are_reported_as_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'style.png'
            Image.new('RGB', (24, 64), 'white').save(path)
            model = torch.nn.Linear(1, 1).train()
            model.image_height = 64
            with patch('historical_swedish.monitoring.render',
                       return_value=(Image.new('L', (40, 64), 237), False)):
                metrics = checkpoint_samples(model, [(path, '')], root/'samples',
                                             'text', 4096, 1, 24, None, 1)
            self.assertEqual(metrics['samples/near_blank_rate'], 1)
            self.assertEqual(metrics['samples/success_rate'], 0)
            self.assertEqual(metrics['samples/eog_rate'], 0)
            record = json.loads((root/'samples/samples.json').read_text())[0]
            self.assertTrue(record['near_blank'])
            self.assertEqual(record['pixel_min'], 237)
            self.assertEqual(record['dark_fraction'], 0)
            self.assertTrue(model.training)

    def test_five_fixed_five_rotating_raw_validation_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, normalized = root/'raw.png', root/'normalized.png'
            Image.new('RGB', (24, 64), (210, 190, 170)).save(raw)
            Image.new('RGB', (24, 64), 'black').save(normalized)
            rows = []
            for split in ('train', 'val'):
                for i in range(21):
                    rows.append(dict(key=f'{split}-{i}', split=split, page_id=split,
                        height=64, width=24, text='four words for style' if i==20 else f'target {i}',
                        raw_path=str(raw), normalized_path=str(normalized)))
            write_jsonl(root/'manifest.jsonl', rows)
            argv = ['train', '--manifest', str(root/'manifest.jsonl'), '--output-dir', str(root/'run'),
                    '--short-steps', '2', '--long-steps', '0', '--batch-size', '2',
                    '--accumulation', '1', '--precision', 'no', '--save-every', '1',
                    '--log-every', '1', '--use', 'raw', '--sample-validation-pairs',
                    '--sample-count', '10', '--sample-fixed-count', '5', '--eval-batches', '1']
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), \
                 patch('historical_swedish.monitoring.render', return_value=(Image.new('L',(40,64),0),True)) as renderer, \
                 patch('sys.argv', argv):
                main()
            first = json.loads((root/'run/stage-0-step-1/samples/samples.json').read_text())
            second = json.loads((root/'run/stage-0-step-2/samples/samples.json').read_text())
            self.assertEqual([r['target_key'] for r in first[:5]], [r['target_key'] for r in second[:5]])
            a, b = {r['target_key'] for r in first[5:]}, {r['target_key'] for r in second[5:]}
            self.assertFalse(a & b)
            self.assertFalse({r['target_key'] for r in first[:5]} & (a | b))
            self.assertEqual([r['preview_group'] for r in first], ['fixed']*5+['rotating']*5)
            self.assertTrue(all(r['style_path']==str(raw) and r['target_path']==str(raw) for r in first+second))
            self.assertEqual(renderer.call_args.args[1].getpixel((0,0)), (210,190,170))
            events=EventAccumulator(str(root/'run/tensorboard')).Reload()
            self.assertEqual([e.step for e in events.Scalars('samples/fixed/eog_rate')],[1,2])
            self.assertEqual([e.step for e in events.Scalars('samples/rotating/eog_rate')],[1,2])
