"""Diffusion objective, sampling, and original-checkpoint compatibility."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from models.diffusion_head import DiffusionHead
from historical_swedish.checkpoints import load_checkpoint, head_options
from historical_swedish.generate import render
from historical_swedish.train import initialize, main
from historical_swedish.common import write_jsonl
from eruku_continuous_inf import DDPCompatibleEmuru
from historical_swedish.adaptation import configure_training
from test_historical import tiny_model


def diffusion_model(height=128, cls=None):
    model = tiny_model(height, cls) if cls is not None else tiny_model(height)
    model.latent_head = 'diffusion'
    model.diffusion_steps = 3
    model.head_config = dict(latent_head='diffusion', diffusion_width=32, diffusion_depth=1,
                             diffusion_steps=3, diffusion_latent_scale=2.0)
    model.diffusion_head = DiffusionHead(16, height // 8, 32, 1, 2.0)
    configure_training(model)
    return model


class DiffusionTests(unittest.TestCase):
    def test_masked_loss_backprop_and_causal_alignment(self):
        model = diffusion_model().train()
        labels = torch.tensor([[2, 0, 2, 2, 1, 1]])
        clean = torch.randn(1, 6, 16)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            losses, predictions = model(clean, labels, ['style'], ['text'])
        self.assertEqual(predictions.shape, (1, 1, 16, 6))
        torch.testing.assert_close(losses['loss'], losses['ce_loss'] + losses['diffusion_loss'])
        losses['loss'].backward()
        for p in (model.query_emb.weight, model.diffusion_head.input.weight, model.t5_to_special.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertIsNone(model.t5_to_vae.weight.grad)
        self.assertTrue(all(not p.requires_grad for p in model.vae.parameters()))
        # Changing the next-column target cannot enter the hidden state used to predict it.
        model.eval()
        changed = clean.clone()
        changed[:, 3:] += 100
        contexts = []
        def capture(target, context):
            contexts.append(context.detach().clone())
            return context.sum() * 0, target.clone()
        with patch.object(model.diffusion_head, 'training_loss', side_effect=capture):
            model(clean, labels, ['style'], ['text'])
            model(changed, labels, ['style'], ['text'])
        # IMG positions are 0, 2, 3; their contexts must all be unchanged.
        torch.testing.assert_close(contexts[0], contexts[1])

    def test_sampler_oracle_and_seeded_cfg(self):
        head = DiffusionHead(4, 2, 16, 1, latent_scale=2)
        context = torch.randn(3, 4)
        target = torch.randn(3, 2)
        def oracle(x, t, context):
            a, s = head.coefficients(t)
            return (a * x - target / 2) / s
        with patch.object(head, 'forward', side_effect=oracle):
            sampled = head.sample(context, steps=7)
        torch.testing.assert_close(sampled, target, atol=1e-5, rtol=1e-5)
        torch.manual_seed(42)
        first = head.sample(context, 3, torch.zeros_like(context), 1.4)
        torch.manual_seed(42)
        second = head.sample(context, 3, torch.zeros_like(context), 1.4)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(first).all())

    def test_linear_initialization_and_roundtrip(self):
        original = tiny_model(128)
        model = diffusion_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            torch.save({'model': original.state_dict()}, path)
            initialize(model, path)
            torch.testing.assert_close(model.query_emb.weight, original.query_emb.weight)
            torch.save({'model': model.state_dict(), 'head_config': model.head_config}, path)
            metadata, state = load_checkpoint(path)
            self.assertEqual(head_options(metadata), model.head_config)
            clone = diffusion_model()
            clone.load_state_dict(state)
            context = torch.randn(1, 1, 16)
            torch.manual_seed(12)
            a = model.predict_next_latent(context)
            torch.manual_seed(12)
            b = clone.predict_next_latent(context)
            torch.testing.assert_close(a, b)

    def test_training_warmup_resume_and_joint_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'line.png'
            Image.new('RGB', (24, 128), 'white').save(image)
            rows = [dict(key=f'{split}{i}', split=split, page_id=split, width=24, height=128,
                         text='two words' if i == 0 else 'four words for style', normalized_path=str(image))
                    for split in ('train', 'val') for i in range(2)]
            write_jsonl(root / 'manifest.jsonl', rows)
            original = tiny_model(128)
            baseline = root / 'baseline.pth'
            torch.save({'model': original.state_dict()}, baseline)
            args = ['train', '--manifest', str(root / 'manifest.jsonl'), '--height', '128',
                    '--short-steps', '0', '--long-steps', '1', '--batch-size', '1', '--accumulation', '1',
                    '--precision', 'no', '--save-every', '1', '--eval-batches', '1', '--no-samples',
                    '--no-tensorboard', '--latent-head', 'diffusion', '--diffusion-width', '32',
                    '--diffusion-depth', '1', '--diffusion-steps', '3', '--diffusion-latent-scale', '2']
            def factory(*a, **kw):
                model = diffusion_model(128, DDPCompatibleEmuru)
                model.diffusion_steps = kw.get('diffusion_steps', 3)
                model.head_config['diffusion_steps'] = model.diffusion_steps
                return model
            def run(extra):
                with patch('historical_swedish.train.DDPCompatibleEmuru',
                           side_effect=factory), \
                     patch('eruku_continuous_inf.AutoencoderKL.from_pretrained', return_value=original.vae), \
                     patch('sys.argv', args + extra):
                    main()
            warmup = root / 'warmup'
            run(['--output-dir', str(warmup), '--init-checkpoint', str(baseline), '--diffusion-head-only'])
            metadata, state = load_checkpoint(warmup / 'stage-1-step-1')
            self.assertEqual(head_options(metadata)['latent_head'], 'diffusion')
            for key, value in original.state_dict().items():
                torch.testing.assert_close(state[key], value.cpu())
            run(['--output-dir', str(warmup), '--resume', str(warmup / 'stage-1-step-1'),
                 '--long-steps', '2', '--diffusion-head-only'])
            joint = root / 'joint'
            run(['--output-dir', str(joint), '--init-checkpoint', str(warmup / 'model.pth')])
            metadata, state = load_checkpoint(joint / 'model.pth')
            clone = diffusion_model()
            clone.load_state_dict(state)
            self.assertEqual(head_options(metadata), clone.head_config)
            self.assertFalse(torch.equal(state['query_emb.weight'], original.query_emb.weight))
            preview_branch = root / 'preview40'
            run(['--output-dir', str(preview_branch), '--resume', str(joint / 'stage-1-step-1'),
                 '--long-steps', '2', '--diffusion-steps', '40'])
            metadata, _ = load_checkpoint(preview_branch / 'model.pth')
            self.assertEqual(head_options(metadata)['diffusion_steps'], 40)
            metadata, _ = load_checkpoint(preview_branch / 'stage-1-step-2')
            self.assertEqual(head_options(metadata)['diffusion_steps'], 40)
            with self.assertRaisesRegex(ValueError, 'diffusion_width'):
                run(['--output-dir', str(root / 'bad-architecture'), '--resume', str(joint / 'stage-1-step-1'),
                     '--long-steps', '2', '--diffusion-width', '64'])

    def test_render_samples_only_image_positions_and_keeps_eog(self):
        model = diffusion_model().eval()
        logits = [torch.tensor([[[0., 0., 9.]]]), torch.tensor([[[0., 9., 0.]]])]
        with patch.object(model.t5_to_special, 'forward', side_effect=logits), \
             patch.object(model.diffusion_head, 'sample', wraps=model.diffusion_head.sample) as sample:
            image, stopped = render(model, Image.new('RGB', (24, 128), 'white'), 'text', cfg_scale=1.3)
        self.assertTrue(stopped)
        self.assertEqual(image.size, (8, 128))
        self.assertEqual(sample.call_count, 1)
        self.assertIsNotNone(sample.call_args.args[2])


if __name__ == '__main__':
    unittest.main()
