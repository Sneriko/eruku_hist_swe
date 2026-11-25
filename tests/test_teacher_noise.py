"""Target-history noise must never become a noisy prediction target."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from eruku_continuous_inf import DDPCompatibleEmuru
from historical_swedish.common import write_jsonl
from historical_swedish.train import main, parser
from test_historical import tiny_model


class TeacherNoiseTests(unittest.TestCase):
    def test_target_only_scale_and_sequence_probability(self):
        model = tiny_model(64).train()
        model.target_noise_probability = .5
        model.target_noise_std_ratio = .05
        # New mode takes priority over legacy all-history noise.
        model.teacher_noise = 1.0
        labels = torch.tensor([[2, 2, 0, 2, 2, 1, 1, -100],
                               [2, 0, 2, 2, 2, 1, -100, -100]])
        clean = torch.arange(128).reshape(2, 8, 8).float()
        clean[0, :2] = 10000  # Style must not inflate the measured target std.
        clean[:, 6:] = 100000  # Nor may padding.
        saved = clean.clone()
        with patch('torch.rand', return_value=torch.tensor([[[0.0]], [[.9]]])), \
             patch('torch.randn_like', side_effect=torch.ones_like):
            noisy = model._teacher_forcing_inputs(clean, labels)
        expected = clean.clone()
        expected[0, 3:5] += clean[0, 3:5].std(correction=0) * .05
        torch.testing.assert_close(noisy, expected)
        torch.testing.assert_close(clean, saved)
        self.assertTrue(torch.equal(noisy[1], clean[1]))

    def test_noise_disabled_in_eval_and_zero_settings_preserve_rng(self):
        model = tiny_model(64)
        clean = torch.randn(1, 6, 8)
        labels = torch.tensor([[2, 2, 0, 2, 2, 1]])
        for training, probability, ratio in [(False, 1, .05), (True, 0, .05), (True, 1, 0)]:
            model.train(training)
            model.target_noise_probability, model.target_noise_std_ratio = probability, ratio
            state = torch.get_rng_state().clone()
            self.assertIs(model._teacher_forcing_inputs(clean, labels), clean)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
        model.train()
        model.target_noise_probability, model.target_noise_std_ratio = 1, .05
        # No target present: style-only sequences must stay clean as well.
        self.assertIs(model._teacher_forcing_inputs(clean, torch.full((1, 6), 2)), clean)

    def test_noise_uses_clean_targets_for_mse_and_bf16_backward(self):
        model = tiny_model(64).train()
        model.target_noise_probability = 1
        model.target_noise_std_ratio = .5
        clean = torch.randn(1, 6, 8)
        labels = torch.tensor([[2, 2, 0, 2, 2, 1]])
        saved = clean.clone()
        # Fix predictions to isolate the regression target from model behavior.
        with patch.object(model.t5_to_vae, 'forward', return_value=torch.zeros_like(clean)), \
             patch.object(model.query_emb, 'forward', wraps=model.query_emb.forward) as projection:
            losses, _ = model(clean, labels, ['four words for style'], ['target text'])
        noisy = projection.call_args.args[0]
        self.assertFalse(torch.equal(noisy[:, 3:5], clean[:, 3:5]))
        torch.testing.assert_close(noisy[:, :3], clean[:, :3])
        torch.testing.assert_close(noisy[:, 5:], clean[:, 5:])
        expected = clean[labels == 2].square().mean()
        torch.testing.assert_close(losses['mse_loss'], expected)
        torch.testing.assert_close(clean, saved)
        model.zero_grad()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses, _ = model(clean, labels, ['four words for style'], ['target text'])
        losses['loss'].backward()
        for parameter in [model.query_emb.weight, model.t5_to_vae.weight, model.t5_to_special.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_noise_cli_validation(self):
        args = ['train', '--manifest', '/missing.jsonl', '--output-dir', '/tmp/unused-noise-test']
        defaults = parser().parse_args(args[1:])
        self.assertEqual(defaults.target_noise_probability, 0)
        self.assertEqual(defaults.target_noise_std_ratio, .05)
        for flag, value in [('--target-noise-probability', '-.1'), ('--target-noise-probability', '1.1'),
                            ('--target-noise-probability', 'nan'), ('--target-noise-std-ratio', '-1'),
                            ('--target-noise-std-ratio', 'inf')]:
            with self.subTest(flag=flag, value=value), patch('sys.argv', args+[flag,value]):
                with self.assertRaisesRegex(ValueError, 'target-noise'):
                    main()

    def test_noise_settings_recorded_and_can_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'line.png'
            Image.new('RGB', (24, 64), 'white').save(image)
            rows = [dict(key=f'{split}{i}', split=split, page_id=split, width=24, height=64,
                         text='two words' if i==0 else 'four words for style', normalized_path=str(image))
                    for split in ('train', 'val') for i in range(2)]
            write_jsonl(root/'manifest.jsonl', rows)
            args = ['train', '--manifest', str(root/'manifest.jsonl'), '--output-dir', str(root/'run'),
                    '--short-steps', '0', '--long-steps', '1', '--batch-size', '1', '--accumulation', '1',
                    '--precision', 'no', '--save-every', '1', '--eval-batches', '1', '--no-samples', '--no-tensorboard']
            models = []
            def factory(*a, **kw):
                model = tiny_model(64, DDPCompatibleEmuru)
                models.append(model)
                return model
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=factory), patch('sys.argv', args):
                main()
            checkpoint = root/'run/stage-1-step-1'
            progress = json.loads((checkpoint/'progress.json').read_text())
            self.assertEqual(progress['args']['target_noise_probability'], 0)
            # A pre-feature checkpoint has no noise arguments at all.
            progress['args'].pop('target_noise_probability')
            progress['args'].pop('target_noise_std_ratio')
            (checkpoint/'progress.json').write_text(json.dumps(progress))
            args += ['--long-steps', '2', '--resume', str(checkpoint),
                     '--target-noise-probability', '.1', '--target-noise-std-ratio', '.05']
            with patch('historical_swedish.train.DDPCompatibleEmuru', side_effect=factory), patch('sys.argv', args):
                main()
            self.assertEqual(models[-1].target_noise_probability, .1)
            self.assertEqual(models[-1].target_noise_std_ratio, .05)
            progress = json.loads((root/'run/stage-1-step-2/progress.json').read_text())
            self.assertEqual(progress['args']['target_noise_probability'], .1)
            self.assertEqual(progress['args']['target_noise_std_ratio'], .05)
