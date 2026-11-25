"""Optional frozen TrOCR supervision for target image predictions."""
import inspect
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel


class FrozenOCRLoss(nn.Module):
    """Backpropagate recognition loss to image latents, never to OCR weights.

    This is an auxiliary training signal, not an independent quality evaluator.
    """
    def __init__(self, model_id, device):
        super().__init__()
        local = Path(model_id).expanduser()
        if local.is_dir():
            source = str(local)
            processor_path = local / 'preprocessor_config.json'
        else:
            from huggingface_hub import hf_hub_download
            source = str(model_id)
            processor_path = Path(hf_hub_download(source, 'preprocessor_config.json'))
        config = json.loads(processor_path.read_text())
        # Newer processor exports nest this dictionary; older Transformers
        # versions otherwise silently fall back to the wrong image dimensions.
        config = config.get('image_processor', config)
        self.image_processor = ViTImageProcessor(**config)
        size = self.image_processor.size
        if not self.image_processor.do_resize or not {'height', 'width'} <= set(size):
            raise ValueError('OCR supervision requires a fixed height/width image processor')
        self.size = (size['height'], size['width'])
        self.tokenizer = AutoTokenizer.from_pretrained(source, extra_special_tokens={})
        self.recognizer = VisionEncoderDecoderModel.from_pretrained(source).to(device)
        self.recognizer.requires_grad_(False).eval()
        self.interpolate_positions = 'interpolate_pos_encoding' in inspect.signature(
            self.recognizer.get_encoder().forward).parameters
        self.register_buffer('mean', torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1))
        self.to(device)
        self.eval()

    def train(self, mode=True):
        # Keep dropout disabled in the frozen teacher, including on mode resets.
        return super().train(False)

    def forward(self, generator, predictions, specials, texts, is_historical=None):
        labels_cpu = specials.detach().cpu()
        selected = ([True] * len(texts) if is_historical is None else
                    is_historical.detach().cpu().tolist())
        images, targets = [], []
        for index, (text, include) in enumerate(zip(texts, selected)):
            if not include:
                continue  # The historical recognizer is not a critic for arbitrary replay fonts.
            sog = (labels_cpu[index] == 0).nonzero().flatten()
            eog = (labels_cpu[index] == 1).nonzero().flatten()
            if len(sog) != 1 or len(eog) < 1 or eog[0] <= sog[0] + 1:
                raise ValueError('OCR loss requires one nonempty target before the first EOG')
            start, end = int(sog[0]) + 1, int(eog[0])
            # Decode ONLY predicted target slices. Reference images and visual
            # delimiters never enter the recognizer as an answer shortcut.
            decoded = generator.vae.decode(predictions[index:index+1, ..., start:end]).sample
            pixels = (decoded.float().clamp(-1, 1) + 1) / 2
            if pixels.shape[1] == 1:
                pixels = pixels.expand(-1, 3, -1, -1)
            elif pixels.shape[1] != 3:
                raise ValueError('OCR supervision requires grayscale or RGB VAE output')
            pixels = F.interpolate(pixels, size=self.size, mode='bilinear',
                                   align_corners=False, antialias=True)
            # Pixels are already in [0,1], unlike the processor's PIL [0,255] inputs.
            if self.image_processor.do_rescale:
                pixels = pixels * (255 * self.image_processor.rescale_factor)
            else:
                pixels = pixels * 255
            if self.image_processor.do_normalize:
                pixels = (pixels - self.mean) / self.std
            images.append(pixels)
            targets.append(text)
        if not images:
            return predictions.new_zeros(())
        tokens = self.tokenizer(targets, padding=True, return_tensors='pt')['input_ids'].to(predictions.device)
        tokens = tokens.masked_fill(tokens == self.tokenizer.pad_token_id, -100)
        kwargs = {'interpolate_pos_encoding': True} if self.interpolate_positions else {}
        return self.recognizer(pixel_values=torch.cat(images), labels=tokens,
                               use_cache=False, return_dict=True, **kwargs).loss
