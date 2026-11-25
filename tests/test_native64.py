"""Native 64px defaults, existing-crop reuse, and explicit OCR export."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from historical_swedish.common import LINE_HEIGHT, MAX_LINE_WIDTH, write_jsonl
from historical_swedish.data import LinePairs, curriculum
from historical_swedish.train import parser
from historical_swedish.generate import main as generate
from test_historical import tiny_model


class Native64Tests(unittest.TestCase):
    def test_defaults_match_pretrained_geometry(self):
        args = parser().parse_args(['--manifest', 'lines.jsonl', '--output-dir', 'run'])
        self.assertEqual((LINE_HEIGHT, args.height), (64, 64))
        self.assertEqual((MAX_LINE_WIDTH, args.max_width), (2048, 2048))
        self.assertEqual([s.max_width for s in curriculum()], [768, 2048])
        self.assertEqual(args.adapter_init, 'strict')
        self.assertFalse(args.adapter_only)

    def test_existing_192_manifest_resizes_before_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, width in [('style', 3000), ('target', 6000)]:
                Image.new('RGB', (width, 192), 'white').save(root/f'{name}.png')
            rows = [dict(key=key, split='train', page_id='page', height=192,
                         width=width, text=text, normalized_path=str(root/f'{path}.png'))
                    for key, width, text, path in
                    [('style', 3000, 'four words in style', 'style'),
                     ('target', 6000, 'full transcription', 'target'),
                     ('too-wide', 6200, 'another full transcription', 'target')]]
            manifest = root/'split.jsonl'
            write_jsonl(manifest, rows)
            before = manifest.read_bytes()
            dataset = LinePairs(manifest, 'train', curriculum()[1])
            self.assertNotIn('too-wide', [r['key'] for r, _ in dataset.pairs])
            index = next(i for i, (r, _) in enumerate(dataset.pairs) if r['key'] == 'target')
            batch = dataset[index]
            self.assertEqual(tuple(batch['style_img'].shape), (3, 64, 1000))
            self.assertEqual(tuple(batch['gen_img'].shape), (3, 64, 2000))
            self.assertEqual(batch['gen_text'], 'full transcription')
            self.assertEqual(manifest.read_bytes(), before)
            with Image.open(root/'target.png') as image:
                self.assertEqual(image.size, (6000, 192))
            Image.new('RGB', (6001, 192), 'white').save(root/'target.png')
            with self.assertRaisesRegex(ValueError, 'geometry differs'):
                dataset[index]

    def test_explicit_192_export_preserves_native_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (24, 64), 'white').save(root/'style.png')
            output = root/'nested'/'export.png'
            argv = ['generate', '--init-checkpoint', 'test-model',
                    '--style-image', str(root/'style.png'), '--text', 'Hej',
                    '--output', str(output), '--output-height', '192', '--device', 'cpu']
            with patch('sys.argv', argv), \
                 patch('historical_swedish.generate.DDPCompatibleEmuru', return_value=tiny_model(64)), \
                 patch('historical_swedish.train.initialize'), \
                 patch('historical_swedish.generate.render', return_value=(Image.new('L', (100, 64)), True)):
                generate()
            with Image.open(output) as image:
                self.assertEqual(image.size, (300, 192))
            report = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual((report['native_width'], report['native_height']), (100, 64))
            self.assertEqual((report['width'], report['height']), (300, 192))
