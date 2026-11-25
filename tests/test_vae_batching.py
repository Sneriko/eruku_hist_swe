"""Batch composition must not change the image representation while preserving paper EOG padding."""
import unittest
from unittest.mock import patch
import torch
from historical_swedish.data import collate_pairs
from test_historical import tiny_model


class VaeBatchingTests(unittest.TestCase):
    def test_latents_match_single_image_with_wider_neighbour(self):
        model = tiny_model(64).eval()
        torch.manual_seed(7)
        short = dict(style_img=torch.rand(3, 64, 32), gen_img=torch.rand(3, 64, 24),
                     style_text='four words in style', gen_text='text')
        long = dict(style_img=torch.rand(3, 64, 80), gen_img=torch.rand(3, 64, 96),
                    style_text='four words in style', gen_text='long text')
        # Remove posterior sampling noise so the test isolates canvas effects.
        def encode(images):
            return model.vae.encode(model.normalize(images)).latent_dist.mode()
        def inputs(samples):
            batch = collate_pairs(samples)
            return model.get_model_inputs(batch['style_img'], batch['gen_img'],
                batch['style_img_width'], batch['gen_img_width'], 256)
        with patch.object(model, '_img_encode', side_effect=encode):
            alone = inputs([short])
            together = inputs([short, long])
        length = alone['specials'].shape[1]
        torch.testing.assert_close(alone['decoder_inputs_embeds'][0],
                                   together['decoder_inputs_embeds'][0, :length])
        torch.testing.assert_close(alone['specials'][0], together['specials'][0, :length])
        self.assertTrue((together['specials'][0, length:] == 1).all())
        self.assertEqual((together['specials'][0] == 1).sum().item(), together['specials'].shape[1] - length + 1)

    def test_explicitly_ignored_labels_do_not_change_loss(self):
        model = tiny_model(64).eval()
        z = torch.randn(1, 6, 8)
        labels = torch.tensor([[2, 2, 0, 2, 2, 1]])
        a, _ = model(z, labels, ['four words in style'], ['text'])
        b, _ = model(torch.nn.functional.pad(z, (0, 0, 0, 8), value=1),
                     torch.nn.functional.pad(labels, (0, 8), value=-100),
                     ['four words in style'], ['text'])
        for key in ('loss', 'ce_loss', 'mse_loss', 'boundary_ce_loss', 'sog_accuracy', 'eog_accuracy'):
            torch.testing.assert_close(a[key], b[key])

    def test_boundary_metrics_distinguish_real_end_from_padding(self):
        model = tiny_model(64).eval()
        labels = torch.tensor([[2, 0, 2, 1, 1]])
        logits = torch.tensor([[[0., 0., 20.], [20., 0., 0.], [0., 0., 20.],
                                [0., 20., 0.], [0., 0., 20.]]])
        with patch.object(model.t5_to_special, 'forward', return_value=logits):
            losses, _ = model(torch.randn(1, 5, 8), labels,
                              ['four words in style'], ['text'])
        self.assertEqual(losses['sog_accuracy'].item(), 1)
        self.assertEqual(losses['eog_accuracy'].item(), 1)
        self.assertLess(losses['boundary_ce_loss'].item(), 1e-5)
        # Paper padding remains supervised even though it is excluded from boundary accuracy.
        self.assertGreater(losses['ce_loss'].item(), 3.9)
        logits[0, 3] = torch.tensor([0., 0., 20.])
        with patch.object(model.t5_to_special, 'forward', return_value=logits):
            losses, _ = model(torch.randn(1, 5, 8), labels,
                              ['four words in style'], ['text'])
        self.assertEqual(losses['eog_accuracy'].item(), 0)
        self.assertEqual(losses['sog_accuracy'].item(), 1)
