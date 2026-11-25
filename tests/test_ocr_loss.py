"""OCR gradients must supervise target images without updating the teacher."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import ViTConfig, TrOCRConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from historical_swedish.ocr_loss import FrozenOCRLoss
from test_historical import tiny_model


class Tokenizer:
    pad_token_id = 0
    def __init__(self):
        self.seen = []
    def __call__(self, texts, **kwargs):
        self.seen = list(texts)
        return {'input_ids': torch.tensor([[3, 4, 1]] * len(texts))}


def recognizer():
    vision = ViTConfig(image_size=32, patch_size=8, hidden_size=16,
                       num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    decoder = TrOCRConfig(vocab_size=32, d_model=16, decoder_layers=1,
                          decoder_attention_heads=2, decoder_ffn_dim=32,
                          dropout=0, attention_dropout=0, activation_dropout=0)
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vision, decoder)
    config.decoder_start_token_id = 2
    config.pad_token_id = 0
    return VisionEncoderDecoderModel(config)


class OCRLossTests(unittest.TestCase):
    def test_only_historical_target_latents_receive_ocr_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'preprocessor_config.json').write_text(json.dumps({
                'image_processor': {'size': {'height': 32, 'width': 32},
                                    'image_mean': [.5]*3, 'image_std': [.5]*3}}))
            tokenizer = Tokenizer()
            with patch('historical_swedish.ocr_loss.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                 patch('historical_swedish.ocr_loss.VisionEncoderDecoderModel.from_pretrained', return_value=recognizer()):
                teacher = FrozenOCRLoss(str(root), torch.device('cpu'))
            teacher.train()
            self.assertFalse(teacher.recognizer.training)
            generator = tiny_model(64)
            predictions = torch.randn(2, 1, 8, 7, requires_grad=True)
            specials = torch.tensor([[2, 2, 0, 2, 2, 1, 1]]*2)
            loss = teacher(generator, predictions, specials,
                           ['historical target', 'replay target'], torch.tensor([True, False]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertEqual(tokenizer.seen, ['historical target'])
            self.assertGreater(predictions.grad[0, ..., 3:5].abs().sum().item(), 0)
            self.assertEqual(predictions.grad[0, ..., :3].abs().sum().item(), 0)
            self.assertEqual(predictions.grad[0, ..., 5:].abs().sum().item(), 0)
            self.assertEqual(predictions.grad[1].abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in teacher.recognizer.parameters()))
            self.assertTrue(all(p.grad is None for p in generator.vae.parameters()))
            with patch.object(teacher.recognizer, 'forward') as call:
                zero = teacher(generator, predictions, specials,
                               ['replay', 'replay'], torch.tensor([False, False]))
                self.assertEqual(zero.item(), 0)
                call.assert_not_called()
