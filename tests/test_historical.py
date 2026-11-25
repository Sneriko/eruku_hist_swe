import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import torch
from PIL import Image
from transformers import T5Config
from diffusers import AutoencoderKL
from eruku_continuous_inf import Emuru, DDPCompatibleEmuru
from historical_swedish.common import write_jsonl, normalize_text, find_page_image
from historical_swedish.data import LinePairs, curriculum, collate_pairs
from historical_swedish.extract_page_lines import extract
from historical_swedish.generate import render


class Tokens(dict):
    def to(self, device):
        return Tokens({k: v.to(device) for k, v in self.items()})


class Tokenizer:
    def add_tokens(self, tokens):
        pass
    def __len__(self):
        return 32
    def __call__(self, texts, **kwargs):
        return Tokens(input_ids=torch.ones(len(texts), 4, dtype=torch.long),
                      attention_mask=torch.ones(len(texts), 4, dtype=torch.long))


def tiny_model(height, cls=Emuru):
    vae = AutoencoderKL(in_channels=3, out_channels=1, latent_channels=1,
        block_out_channels=(8, 8, 8, 8), norm_num_groups=4,
        down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4)
    config = T5Config(d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8)
    with patch('eruku_continuous_inf.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
         patch('eruku_continuous_inf.T5Config.from_pretrained', return_value=config), \
         patch('eruku_continuous_inf.AutoencoderKL.from_pretrained', return_value=vae):
        return cls(ocr_checkpoint=None, image_height=height)


class HistoricalTests(unittest.TestCase):
    def test_latent_training_and_budget(self):
        for height in (64, 192):
            model = tiny_model(height)
            model.train()
            self.assertFalse(model.vae.training)
            images = torch.ones(2, 3, height, 32)
            batch = model.get_model_inputs(images, images, [16, 32], [24, 32], 80)
            self.assertEqual(batch['decoder_inputs_embeds'].shape, (2, 10, height//8))
            self.assertEqual(batch['specials'][0].tolist(), [2, 2, 0, 2, 2, 2, 1, 1, 1, 1])
            losses, pred = model(batch['decoder_inputs_embeds'], batch['specials'], ['å ä', 'ö'], ['ſwensk', 'text'])
            self.assertTrue(torch.isfinite(losses['loss']))
            losses['loss'].backward()
            self.assertIsNotNone(model.query_emb.weight.grad)
            self.assertEqual(pred.shape[2], height//8)
            with self.assertRaisesRegex(ValueError, 'refusing'):
                model.get_model_inputs(images, images, [32, 32], [32, 32], 64)

    def test_bf16_special_token_forward_backward(self):
        # Autocast makes Linear outputs bf16 while embedding weights stay fp32.
        # Exercise both delimiter assignments and verify their gradients survive casts.
        model = tiny_model(192).train()
        model.gradient_checkpointing_enable({"use_reentrant": False})
        latents = torch.randn(2, 5, 24)
        specials = torch.tensor([[2, 0, 2, 1, 1], [2, 2, 0, 2, 1]])
        for training in (True, False):
            model.train(training)
            model.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                losses, _ = model(latents, specials, ['fyra ord i stilen']*2, ['åäö text']*2)
            self.assertTrue(torch.isfinite(losses['loss']))
            if training:
                losses['loss'].backward()
                for token in (model.sog, model.eog, model.sos):
                    self.assertIsNotNone(token.weight.grad)
                    self.assertTrue(torch.isfinite(token.weight.grad).all())

    def test_generation_excludes_special_tokens_and_style(self):
        model = tiny_model(192).eval()
        # Force 2 image slices and then EOG.
        logits = [torch.tensor([[[0., 0., 9.]]]), torch.tensor([[[0., 0., 9.]]]), torch.tensor([[[0., 9., 0.]]])]
        with patch.object(model.t5_to_special, 'forward', side_effect=logits):
            image, stopped = render(model, Image.new('RGB', (24, 192), 'white'), 'åäö', max_width=2048, cfg_scale=1)
        self.assertEqual(image.size, (16, 192))
        self.assertTrue(stopped)

    def test_generation_over_2000_pixels(self):
        model = tiny_model(192).eval()
        with patch.object(model.T5, 'forward', return_value=SimpleNamespace(logits=torch.zeros(1, 1, 16))),              patch.object(model.t5_to_special, 'forward', return_value=torch.tensor([[[0., 0., 9.]]])):
            image, stopped = render(model, Image.new('RGB', (24, 192), 'white'), 'lång text', max_width=2056, cfg_scale=1)
        self.assertEqual(image.size, (2056, 192))
        self.assertFalse(stopped)

    def test_page_extraction_and_pair_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (900, 160), 'white').save(root/'p.png')
            lines = []
            for i in range(2):
                words = ''.join(f'<Word id="w{i}{j}"><Coords points="{10+j*100},{10+i*70} {90+j*100},{10+i*70} {90+j*100},{50+i*70} {10+j*100},{50+i*70}"/><TextEquiv><Unicode>{t}</Unicode></TextEquiv></Word>' for j, t in enumerate(['åäö', 'ſwensk', 'text', 'här']))
                lines.append(f'<TextLine id="l{i}"><Coords points="10,{10+i*70} 400,{10+i*70} 400,{50+i*70} 10,{50+i*70}"/><TextEquiv><Unicode>åäö ſwensk text här</Unicode></TextEquiv>{words}</TextLine>')
            (root/'p.xml').write_text('<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"><Page imageFilename="p.png">'+''.join(lines)+'</Page></PcGts>')
            out = root/'out'
            extract(SimpleNamespace(root=root, output=out, metadata_csv=None, xml_glob='*.xml', strict=True,
                unicode_form='NFC', min_chars=2, padding=0, height=192, max_width=6144, min_width=24, word_spans=True))
            rows = [json.loads(s) for s in (out/'lines.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['height'] == 192 for r in rows))
            self.assertTrue(any(r['width'] > 1000 for r in rows))
            for row in rows:
                row['split'] = 'train'
            write_jsonl(out/'split.jsonl', rows)
            dataset = LinePairs(out/'split.jsonl', 'train', curriculum()[0])
            for target, candidates in dataset.pairs:
                self.assertTrue(all(len(s['text'].split()) >= 3 for s in candidates))
                self.assertTrue(all(target['source_line_id'] != s['source_line_id'] for s in candidates))
            batch = collate_pairs([dataset[0], dataset[-1]])
            self.assertEqual(batch['style_img'].shape[2], 64)
            self.assertEqual(batch['gen_img'].shape[-1] % 8, 0)
            self.assertIn('ſ', batch['gen_text'][0])
            self.assertEqual(normalize_text('a\u030a'), 'å')

    def test_page_image_folder_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for layout in ('beside-page', 'image'):
                root = base / layout
                (root / 'page').mkdir(parents=True)
                image_dir = root if layout == 'beside-page' else root / 'image'
                image_dir.mkdir(exist_ok=True)
                image = image_dir / 'p.png'
                Image.new('RGB', (20, 20), 'white').save(image)
                xml = root / 'page' / 'p.xml'
                for filename in ('p.png', None):
                    self.assertEqual(find_page_image(xml, filename), image.resolve())
                if layout == 'image':
                    for filename in ('image/p.png', '../image/p.png'):
                        self.assertEqual(find_page_image(xml, filename), image.resolve())
                    renamed = image_dir / 'different.png'
                    image.rename(renamed)
                    self.assertEqual(find_page_image(xml, 'different.png'), renamed.resolve())

    def test_multiple_roots_with_duplicate_page_names(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = [base/'first'/'dataset', base/'second'/'dataset']
            for root in roots:
                root.mkdir(parents=True)
                Image.new('RGB', (200, 60), 'white').save(root/'p.png')
                (root/'p.xml').write_text('<PcGts><Page imageFilename="p.png"><TextLine id="l1">'
                    '<Coords points="0,0 190,0 190,50 0,50"/><TextEquiv><Unicode>fyra ord i stilen</Unicode>'
                    '</TextEquiv></TextLine></Page></PcGts>')
            args = SimpleNamespace(root=roots + [roots[0], base/'first'], output=base/'out', metadata_csv=None,
                xml_glob='*.xml', strict=True, unicode_form='NFC', min_chars=2, padding=0,
                height=192, max_width=6144, min_width=24, word_spans=False)
            extract(args)
            rows = [json.loads(s) for s in (args.output/'lines.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)  # repeated/overlapping roots do not duplicate pages
            for field in ('key', 'page_id', 'document_id', 'pseudo_writer_id', 'dataset_id'):
                self.assertEqual(len({r[field] for r in rows}), 2, field)
            self.assertEqual({r['dataset_root'] for r in rows}, {str(r) for r in roots})
            self.assertTrue(all(Path(r['normalized_path']).is_file() for r in rows))

    def test_train_checkpoint_resume(self):
        from historical_swedish.train import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'line.png'
            Image.new('RGB', (24, 64), 'white').save(image)
            rows = [{'key': f'{split}{i}', 'split': split, 'page_id': split, 'height': 64,
                     'width': 24, 'text': 'åäö text' if i == 0 else 'fyra ord i stilen', 'normalized_path': str(image)}
                    for split in ('train', 'val') for i in range(2)]
            write_jsonl(root/'lines.jsonl', rows)
            argv = ['train', '--manifest', str(root/'lines.jsonl'), '--output-dir', str(root/'run'),
                    '--short-steps', '1', '--long-steps', '1', '--batch-size', '2',
                    '--accumulation', '1', '--precision', 'no', '--save-every', '1',
                    '--no-tensorboard', '--no-samples']
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), patch('sys.argv', argv):
                main()
            from test_replay import write_shard, sample
            write_shard(root/'replay.tar', [sample(width=8)])
            replay_argv = argv + ['--replay-train-pattern', str(root/'replay.tar'),
                                  '--historical-probability', '0.00001']
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), patch('sys.argv', replay_argv):
                main()
            argv = replay_argv
            self.assertTrue((root/'run/model.pth').exists())
            progress = json.loads((root/'run/stage-1-step-1/progress.json').read_text())
            self.assertEqual((progress['stage'], progress['step']), (1, 1))
            self.assertEqual(progress['input_pipeline_version'], 3)
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), patch('sys.argv', argv + ['--resume', str(root/'run/stage-0-step-1')]):
                main()

            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), patch('sys.argv', argv + ['--long-steps', '2', '--resume', str(root/'run/stage-1-step-1')]):
                main()
            extended = json.loads((root/'run/stage-1-step-2/progress.json').read_text())
            self.assertEqual(extended['step'], 2)
            self.assertEqual(extended['stages'][-1]['steps'], 2)
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), patch('sys.argv', argv + ['--long-steps', '2', '--max-width', '2056', '--resume', str(root/'run/stage-1-step-1')]):
                with self.assertRaisesRegex(ValueError, 'curriculum/height differ'):
                    main()

            # Old optimizer checkpoints used a different VAE/loss contract.
            old_path = root/'run/stage-0-step-1/progress.json'
            old_progress = json.loads(old_path.read_text())
            old_progress.pop('input_pipeline_version')
            old_path.write_text(json.dumps(old_progress))
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=lambda *a, **kw: tiny_model(64, DDPCompatibleEmuru)), patch('sys.argv', argv + ['--resume', str(old_path.parent)]):
                with self.assertRaisesRegex(ValueError, 'earlier VAE/padding'):
                    main()

    def test_three_word_style_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory)/'lines.jsonl'
            rows = [dict(key=str(i), split='train', page_id='page', height=64,
                         width=100, text=text, normalized_path='unused')
                    for i, text in enumerate(['target', 'three style words', 'two words'])]
            write_jsonl(manifest, rows)
            pairs = LinePairs(manifest, 'train', curriculum()[1])
            target, candidates = next((t, c) for t, c in pairs.pairs if t['key']=='0')
            self.assertEqual([c['text'] for c in candidates], ['three style words'])

    def test_curricula(self):
        short, long = curriculum()
        self.assertEqual((short.steps, long.steps), (65000, 5000))
        self.assertEqual(long.style_dropout, 0.1)
        self.assertEqual(short.style_words, (3, 8))
        self.assertEqual(long.style_words, (3, 8))
        short, long = curriculum('emuru')
        self.assertEqual((short.steps, long.steps), (70000, 250))
        self.assertEqual((short.max_width, long.max_width), (768, 2048))
        with self.assertRaises(ValueError):
            curriculum(max_width=760)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
