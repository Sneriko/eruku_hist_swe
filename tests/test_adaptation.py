"""Verify height transfer preserves channel layout and warm-up freezes the backbone."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from historical_swedish.generate import render
import torch
from historical_swedish.adaptation import configure_training, optimizer_groups, resize_adapter
from historical_swedish.train import initialize
from test_historical import tiny_model


class AdaptationTests(unittest.TestCase):
    def test_channel_layout_and_projection_scale(self):
        # Constant vertical signals should keep each channel separate and preserve
        # the input projection response when going from height 2 to height 6.
        source = torch.tensor([[1., 1., 4., 4.]])
        target = torch.empty(1, 12)
        actual = resize_adapter('query_emb.weight', source, target, 2)
        torch.testing.assert_close(actual @ torch.ones(12), source @ torch.ones(4))
        output = resize_adapter('t5_to_vae.bias', source[0], torch.empty(12), 2)
        torch.testing.assert_close(output, torch.tensor([1.]*6 + [4.]*6))

    def test_transfer_and_adapter_only_backward(self):
        original, model = tiny_model(64), tiny_model(192)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'original.pth'
            torch.save({'model': original.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "Checkpoint image adapters differ"):
                initialize(model, path)
            initialize(model, path, adapter_init="interpolate")
        torch.testing.assert_close(model.T5.shared.weight, original.T5.shared.weight)
        torch.testing.assert_close(model.query_emb.bias, original.query_emb.bias)
        expected = resize_adapter('t5_to_vae.weight', original.t5_to_vae.weight,
                                  model.t5_to_vae.weight, 1)
        torch.testing.assert_close(model.t5_to_vae.weight, expected)
        configure_training(model, True)
        model.gradient_checkpointing_enable({'use_reentrant': False})
        frozen_before = model.T5.shared.weight.detach().clone()
        adapter_before = model.t5_to_vae.weight.detach().clone()
        optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-3))
        losses, _ = model(torch.randn(1, 5, 24), torch.tensor([[2, 2, 0, 2, 1]]),
                          ['four words in style'], ['text'])
        losses['loss'].backward()
        self.assertIsNotNone(model.query_emb.weight.grad)
        self.assertIsNotNone(model.t5_to_vae.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.T5.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.vae.parameters()))
        optimizer.step()
        torch.testing.assert_close(model.T5.shared.weight, frozen_before)
        self.assertFalse(torch.equal(model.t5_to_vae.weight, adapter_before))
        configure_training(model, False)
        groups = optimizer_groups(model, 2e-5, 1e-4)
        self.assertEqual([g['lr'] for g in groups], [1e-4, 2e-5])
        self.assertTrue(model.T5.shared.weight.requires_grad)
        self.assertFalse(model.vae.encoder.conv_in.weight.requires_grad)

    def test_preview_limit_independent_of_style_width(self):
        model = tiny_model(192).eval()
        with patch.object(model.t5_to_special, 'forward',
                          return_value=torch.tensor([[[0., 0., 9.]]])):
            image, stopped = render(model, Image.new('RGB', (64, 192), 'white'),
                                    'text', max_width=16, cfg_scale=1, style_max_width=64)
        self.assertEqual(image.size, (16, 192))
        self.assertFalse(stopped)

    def test_frozen_decoder_keeps_prior_but_allows_encoder_and_adapter_learning(self):
        model = tiny_model(64)
        configure_training(model, freeze_decoder=True)
        model.gradient_checkpointing_enable({'use_reentrant': False})
        model.train()
        self.assertFalse(model.T5.decoder.training)
        self.assertTrue(model.T5.encoder.training)
        self.assertTrue(model.T5.shared.weight.requires_grad)
        weight = model.T5.decoder.block[0].layer[0].SelfAttention.q.weight
        before = weight.detach().clone()
        optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-3))
        losses, _ = model(torch.randn(1, 5, 8), torch.tensor([[2, 2, 0, 2, 1]]),
                          ['four words in style'], ['text'])
        losses['loss'].backward()
        self.assertIsNone(weight.grad)
        self.assertIsNotNone(model.T5.shared.weight.grad)
        self.assertIsNotNone(model.query_emb.weight.grad)
        self.assertTrue(torch.isfinite(model.query_emb.weight.grad).all())
        optimizer.step()
        torch.testing.assert_close(weight, before)
        model.eval()
        model.train()
        self.assertFalse(model.T5.decoder.training)
        configure_training(model, freeze_decoder=False)
        self.assertTrue(model.T5.decoder.training)
        self.assertTrue(weight.requires_grad)

    def test_frozen_decoder_ffn_preserves_feedforward_but_trains_attention(self):
        model = tiny_model(64)
        configure_training(model, freeze_decoder_ffn=True)
        model.gradient_checkpointing_enable({'use_reentrant': False})
        model.train()
        block = model.T5.decoder.block[0]
        feedforward = block.layer[2].DenseReluDense.wi.weight
        before = feedforward.detach().clone()
        attention = block.layer[0].SelfAttention.q.weight
        self.assertTrue(model.T5.decoder.training)
        self.assertFalse(block.layer[2].training)
        self.assertTrue(block.layer[0].training)
        optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-3))
        losses, _ = model(torch.randn(1, 5, 8), torch.tensor([[2, 2, 0, 2, 1]]),
                          ['four words in style'], ['text'])
        losses['loss'].backward()
        self.assertIsNone(feedforward.grad)
        self.assertIsNotNone(attention.grad)
        self.assertTrue(torch.isfinite(attention.grad).all())
        self.assertIsNotNone(model.query_emb.weight.grad)
        optimizer.step()
        torch.testing.assert_close(feedforward, before)
        model.eval()
        model.train()
        self.assertFalse(block.layer[2].training)
        configure_training(model)
        self.assertTrue(block.layer[2].training)
        self.assertTrue(feedforward.requires_grad)
