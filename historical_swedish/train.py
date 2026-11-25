"""Train Eruku on PAGE-derived lines with a resumable curricula."""
import argparse
import math
from dataclasses import asdict
from pathlib import Path
import torch
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from eruku_continuous_inf import DDPCompatibleEmuru
from .common import LINE_HEIGHT, MAX_LINE_WIDTH
from .checkpoints import load_checkpoint, head_options, HEAD_DEFAULTS
from .adaptation import resize_adapter, configure_training, optimizer_groups
from .data import LinePairs, collate_pairs, curriculum
from .replay import ReplayMix, resolve_shards, REPLAY_SAMPLING_VERSION
from .ocr_loss import FrozenOCRLoss
from .monitoring import isolated_rng, select_styles, select_validation_styles, checkpoint_samples


METRICS = ('loss', 'ce_loss', 'mse_loss', 'diffusion_loss', 'ocr_loss', 'boundary_ce_loss', 'sog_accuracy', 'eog_accuracy')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--replay-train-pattern', help='Local paired FontSquare .tar glob or .txt shard list')
    p.add_argument('--historical-probability', type=float, default=0.9,
                   help='Historical sample probability when replay is enabled (default: 0.9)')
    p.add_argument('--allow-replay-ratio-change', action='store_true',
                   help='Explicitly permit changing historical probability on resume while retaining optimizer state')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--curriculum', choices=['eruku', 'emuru', 'three-stage'], default='eruku')
    p.add_argument('--short-steps', type=int)
    p.add_argument('--medium-steps', type=int)
    p.add_argument('--long-steps', type=int)
    p.add_argument('--max-width', type=int, default=MAX_LINE_WIDTH)
    p.add_argument('--height', type=int, default=LINE_HEIGHT)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--ocr-model', help='Optional HF/local historical TrOCR checkpoint for frozen target-image supervision')
    p.add_argument('--ocr-loss-weight', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--freeze-decoder-ffn', action='store_true', help='Preserve pretrained decoder feed-forward blocks and their layer norms; attention remains trainable')
    p.add_argument('--freeze-decoder', action='store_true', help='Preserve pretrained decoder blocks and final norm while training text encoder, adapters and markers')
    p.add_argument('--adapter-only', action='store_true', help='Train only image adapters; inspect previews before a separate full fine-tuning run')
    p.add_argument('--adapter-lr', type=float, help='Image adapter learning rate (defaults to --lr)')
    p.add_argument('--adapter-init', choices=['strict', 'interpolate', 'random'], default='strict')
    p.add_argument('--sample-max-width', type=int, help='Preview output width limit, independent of reference/training widths')
    p.add_argument('--target-noise-probability', type=float, default=0.0,
                   help='Fraction of training sequences with noise on target-image history only (default: disabled)')
    p.add_argument('--target-noise-std-ratio', type=float, default=0.05,
                   help='Noise std as a fraction of each clean target sequence latent std; not an absolute std (default: 0.05)')
    p.add_argument('--text-dropout', type=float, default=0.05)
    p.add_argument('--t5-config', default='google-t5/t5-large')
    p.add_argument('--latent-head', choices=['linear', 'diffusion'], default='linear')
    p.add_argument('--diffusion-width', type=int, default=256)
    p.add_argument('--diffusion-depth', type=int, default=3)
    p.add_argument('--diffusion-steps', type=int, default=20, help='Denoising steps for previews/inference')
    p.add_argument('--diffusion-latent-scale', type=float, default=1.0, help='Fixed divisor for VAE latents inside the diffusion head')
    p.add_argument('--diffusion-head-only', action='store_true', help='Warm up only the new denoiser; freeze the entire original model')
    p.add_argument('--vae', default='blowing-up-groundhogs/emuru_vae')
    p.add_argument('--resume', type=Path)
    p.add_argument('--init-checkpoint', help='Local .pth file or Hugging Face Eruku model ID')
    p.add_argument('--init-checkpoint-filename', default='000073688.pth',
                   help='Weight filename for a Hugging Face model ID (default: 000073688.pth)')
    p.add_argument('--init-checkpoint-revision', default='main', help='Hugging Face branch, tag or commit')
    p.add_argument('--pair-by', choices=['page_id', 'pseudo_writer_id'], default='page_id')
    p.add_argument('--use', choices=['raw', 'normalized'], default='normalized')
    p.add_argument('--precision', choices=['no', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=20)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--no-tensorboard', action='store_true')
    p.add_argument('--sample-style-dir', type=Path,
                   default=Path('/mnt/scratch/emuru_eruku_data/data/example_inference'))
    p.add_argument('--sample-validation-pairs', action='store_true', help='Preview fixed held-out historical pairs with actual target text instead of the constant modern prompt')
    p.add_argument('--sample-split', choices=['test', 'val', 'style'], default='test', help='Checkpoint preview source; default: test')
    p.add_argument('--sample-count', type=int, default=50)
    p.add_argument('--sample-fixed-count', type=int, help='Fixed validation previews; remaining previews rotate at each checkpoint (default: all fixed)')
    p.add_argument('--sample-text', default='Hej, jag heter Erik och jag tränar Eruku')
    p.add_argument('--sample-cfg-scale', type=float, default=1.25)
    p.add_argument('--sample-cfg-scales', type=float, nargs='+')
    p.add_argument('--sample-seeds', type=int, nargs='+')
    p.add_argument('--sample-teacher-forced', action='store_true')
    p.add_argument('--sample-alt-vae', help='Optional second VAE decoder for the same generated latents')
    p.add_argument('--sample-ocr', help='Optional local/HF recognizer for checkpoint CER (runs on CPU)')
    p.add_argument('--no-samples', action='store_true')
    p.add_argument('--seed', type=int, default=24)
    return p


def initialize(model, path, filename='000073688.pth', revision='main', adapter_init='strict'):
    local = Path(path).expanduser()
    if local.is_file() or local.is_dir():
        path = local
    elif local.is_absolute() or str(path).startswith(('.', '~')) or str(path).endswith(('.pth', '.bin', '.safetensors')):
        raise FileNotFoundError(f'Initialization checkpoint does not exist: {local}')
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=str(path), filename=filename, revision=revision)
        print(f'Initializing Hugging Face checkpoint {path}')
    metadata, state = load_checkpoint(path)
    source_head = head_options(metadata)
    if any(key.startswith('diffusion_head.') for key in state):
        if model.latent_head != 'diffusion':
            raise ValueError('Diffusion checkpoint requires --latent-head diffusion')
        for key in ('diffusion_width', 'diffusion_depth', 'diffusion_latent_scale'):
            if source_head[key] != model.head_config[key]:
                raise ValueError(f'Initialization diffusion setting differs: {key}')
    current = model.state_dict()
    mismatches = {k for k in state if k in current and state[k].shape != current[k].shape}
    allowed = {'query_emb.weight', 't5_to_vae.weight', 't5_to_vae.bias'}
    if mismatches - allowed:
        raise ValueError(f'Unexpected incompatible weights: {mismatches - allowed}')
    if mismatches and adapter_init == 'strict':
        raise ValueError('Checkpoint image adapters differ from the requested height. '
                         'For native 64px training, initialize from the original Eruku checkpoint, '
                         'not an experimental 192px checkpoint.')
    for k in mismatches:
        if adapter_init == 'interpolate':
            state[k] = resize_adapter(k, state[k], current[k], model.latent_channels)
        else:
            del state[k]
    result = model.load_state_dict(state, strict=False)
    # A linear checkpoint deliberately has no denoiser weights.
    new_head = {key for key in current if key.startswith('diffusion_head.')} if source_head['latent_head'] == 'linear' else set()
    unexpected_missing = set(result.missing_keys) - allowed - new_head
    if unexpected_missing:
        raise ValueError(f'Checkpoint missing model parameters: {unexpected_missing}')
    print(f'Initialized checkpoint; height adapter initialization={adapter_init}: {sorted(mismatches)}; unused keys: {result.unexpected_keys}')


def main():
    args = parser().parse_args()
    if min(args.batch_size, args.accumulation, args.save_every, args.eval_batches) <= 0 or args.workers < 0:
        raise ValueError('batch-size, accumulation, save-every, eval-batches must be positive; workers nonnegative')
    if not 0 <= args.text_dropout <= 1 or args.height <= 0 or args.height % 8:
        raise ValueError('Invalid text dropout or image height')
    if args.log_every <= 0 or args.sample_count < 10 or args.sample_cfg_scale < 0:
        raise ValueError('log-every must be positive, sample-count at least 10, and sample-cfg-scale nonnegative')
    if args.sample_validation_pairs:
        args.sample_split = 'val'
    if args.sample_cfg_scales and (len(set(args.sample_cfg_scales)) != len(args.sample_cfg_scales)
            or any(not math.isfinite(cfg) or cfg < 0 for cfg in args.sample_cfg_scales)):
        raise ValueError('sample-cfg-scales must be finite, distinct and nonnegative')
    if args.sample_seeds and len(set(args.sample_seeds)) != len(args.sample_seeds):
        raise ValueError('sample-seeds must be distinct')
    if args.sample_teacher_forced and args.sample_split == 'style':
        raise ValueError('Teacher-forced previews require paired samples')
    paired_samples = args.sample_split != 'style'
    if args.sample_fixed_count is not None:
        if not paired_samples or not 0 <= args.sample_fixed_count <= args.sample_count:
            raise ValueError('sample-fixed-count requires test/validation-pair previews and must be between 0 and sample-count')
    if not 0 <= args.target_noise_probability <= 1:
        raise ValueError('target-noise-probability must be in [0, 1]')
    if not math.isfinite(args.target_noise_std_ratio) or args.target_noise_std_ratio < 0:
        raise ValueError('target-noise-std-ratio must be finite and nonnegative')
    if args.ocr_loss_weight < 0 or bool(args.ocr_model) != (args.ocr_loss_weight > 0):
        raise ValueError('Set both --ocr-model and a positive --ocr-loss-weight, or neither')
    if args.lr <= 0 or (args.adapter_lr is not None and args.adapter_lr <= 0):
        raise ValueError('Learning rates must be positive')
    if args.sample_max_width is not None and (args.sample_max_width < 8 or args.sample_max_width % 8):
        raise ValueError('sample-max-width must be a positive multiple of 8')
    if (args.freeze_decoder or args.freeze_decoder_ffn) and not (args.init_checkpoint or args.resume):
        raise ValueError('Freezing the decoder requires pretrained initialization or resume')
    if args.adapter_only and not (args.init_checkpoint or args.resume):
        raise ValueError('Adapter warm-up requires --init-checkpoint or --resume')
    if args.resume and args.init_checkpoint:
        raise ValueError('Choose either --resume or --init-checkpoint')
    if not 0 < args.historical_probability <= 1:
        raise ValueError('historical-probability must be in (0, 1]')
    if args.diffusion_steps < 1 or args.diffusion_width < 2 or args.diffusion_depth < 1:
        raise ValueError('Diffusion steps/depth must be positive and width at least 2')
    if not math.isfinite(args.diffusion_latent_scale) or args.diffusion_latent_scale <= 0:
        raise ValueError('diffusion-latent-scale must be finite and positive')
    if args.diffusion_head_only and (args.latent_head != 'diffusion' or not (args.init_checkpoint or args.resume)):
        raise ValueError('diffusion-head-only requires diffusion and pretrained initialization/resume')
    if args.diffusion_head_only and (args.adapter_only or args.freeze_decoder or args.freeze_decoder_ffn):
        raise ValueError('diffusion-head-only cannot be combined with other freezing modes')
    if args.latent_head == 'diffusion' and args.ocr_loss_weight:
        raise ValueError('OCR supervision is not supported for this diffusion experiment')
    shards = resolve_shards(args.replay_train_pattern) if args.replay_train_pattern else None
    if not shards and args.historical_probability != 0.9:
        raise ValueError('historical-probability requires --replay-train-pattern')
    stages = curriculum(args.curriculum, args.short_steps, args.long_steps, args.max_width,
                        medium_steps=args.medium_steps)
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation, mixed_precision=args.precision,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    styles = select_styles(args.sample_style_dir, args.sample_count, args.height, args.max_width,
                           args.seed) if not args.no_samples and not paired_samples else []
    SummaryWriter = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
    set_seed(args.seed)
    model = DDPCompatibleEmuru(args.t5_config, args.vae, ocr_checkpoint=None, image_height=args.height,
                               **{key: getattr(args, key) for key in HEAD_DEFAULTS})
    model.target_noise_probability = args.target_noise_probability
    model.target_noise_std_ratio = args.target_noise_std_ratio
    if args.target_noise_probability > 0:
        accelerator.print(f'Target-history teacher noise: probability={args.target_noise_probability}; '
                          f'std={args.target_noise_std_ratio} x per-sequence clean target latent std; '
                          'style/markers/targets stay clean; training only (replaces legacy teacher noise)')
    model.drop_text = True
    model.dropout_probability = args.text_dropout
    model.gradient_checkpointing_enable({"use_reentrant": False})
    if args.init_checkpoint:
        initialize(model, args.init_checkpoint, args.init_checkpoint_filename, args.init_checkpoint_revision, args.adapter_init)
        # Eruku checkpoints include VAE weights; restore the explicitly configured VAE.
        model.vae = type(model.vae).from_pretrained(args.vae)
        model.set_training(model.vae, False)
        accelerator.print(f"Using frozen VAE: {args.vae}")
    configure_training(model, args.adapter_only, args.freeze_decoder, args.freeze_decoder_ffn)
    accelerator.print('Training image adapters only' if args.adapter_only else
                      'Training text encoder, image adapters and markers; decoder blocks and VAE frozen' if args.freeze_decoder else
                      'Training attention, text encoder, adapters and markers; decoder feed-forward blocks and VAE frozen' if args.freeze_decoder_ffn else
                      'Training transformer and image adapters; VAE frozen')
    if args.diffusion_head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('diffusion_head.'))
        accelerator.print('Diffusion warm-up: training denoiser only; original model frozen')
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.lr, args.adapter_lr), weight_decay=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    start_stage = start_step = 0
    if args.resume:
        import json
        progress = json.loads((args.resume / 'progress.json').read_text())
        expected_stages = [asdict(s) | {'style_words': list(s.style_words), 'target_words': list(s.target_words)} for s in stages]
        saved_stages = progress['stages']
        # Constant LR has no schedule to rescale. Extending only the final stage
        # preserves previous-stage offsets and allows a reviewed pilot to continue.
        final_extension = (len(saved_stages) == len(expected_stages)
            and progress['stage'] == len(expected_stages) - 1
            and saved_stages[:-1] == expected_stages[:-1]
            and {k: v for k, v in saved_stages[-1].items() if k != 'steps'} ==
                {k: v for k, v in expected_stages[-1].items() if k != 'steps'}
            and expected_stages[-1]['steps'] >= saved_stages[-1]['steps'])
        if (saved_stages != expected_stages and not final_extension) or progress['height'] != args.height:
            raise ValueError('Resume curriculum/height differ from checkpoint')
        if progress.get('input_pipeline_version', 1) != 3:
            raise ValueError('Checkpoint uses an earlier VAE/padding training pipeline; start a fresh run from the original checkpoint')
        previous = progress.get('args', {})
        for key, default in HEAD_DEFAULTS.items():
            if previous.get(key, default) != getattr(args, key):
                if key == 'diffusion_steps':
                    accelerator.print(f'Diffusion preview sampling steps changed on resume: '
                                      f'{previous.get(key, default)} -> {args.diffusion_steps}; '
                                      'training objective and optimizer state are unchanged')
                else:
                    raise ValueError(f'Resume architecture differs: {key}; initialize a new run instead')
        if previous.get('diffusion_head_only', False) != args.diffusion_head_only:
            raise ValueError('Resume warm-up mode differs; use --init-checkpoint for joint fine-tuning')
        previous_noise = (previous.get('target_noise_probability', 0.0),
                          previous.get('target_noise_std_ratio', 0.05))
        current_noise = (args.target_noise_probability, args.target_noise_std_ratio)
        if previous_noise != current_noise:
            accelerator.print(f'Target-history noise changed on resume: {previous_noise} -> {current_noise}; '
                              'current settings apply from the next update')
        if previous.get('replay_train_pattern') != args.replay_train_pattern:
            raise ValueError('Resume replay source differs from checkpoint')
        previous_probability = previous.get('historical_probability', 0.9)
        if args.replay_train_pattern and previous_probability != args.historical_probability:
            if not args.allow_replay_ratio_change:
                raise ValueError('Resume replay ratio differs from checkpoint; use --allow-replay-ratio-change for an intentional experiment')
            accelerator.print(f'Historical sampling probability changed on resume: '
                              f'{previous_probability} -> {args.historical_probability}; '
                              'optimizer state is retained and the new mixture applies from the next update')
        if previous.get('use', 'normalized') != args.use:
            raise ValueError('Resume image mode differs; initialize a new run for raw/normalized comparisons')
        if shards and progress.get('replay_sampling_version', 1) != REPLAY_SAMPLING_VERSION:
            raise ValueError('Resume replay sampling differs; start a new run for shuffled historical coverage')
        if previous.get('ocr_model') != args.ocr_model or previous.get('ocr_loss_weight', 0.0) != args.ocr_loss_weight:
            raise ValueError('Resume OCR supervision differs; initialize a new run instead')
        if previous.get('freeze_decoder_ffn', False) != args.freeze_decoder_ffn:
            raise ValueError('Resume decoder feed-forward freezing differs; initialize a new run instead')
        if previous.get('freeze_decoder', False) != args.freeze_decoder:
            raise ValueError('Resume decoder-freezing mode differs; initialize a new run instead')
        if previous.get('adapter_only', False) != args.adapter_only:
            raise ValueError('Resume adapter-only mode differs; use --init-checkpoint for the next phase')
        if (previous.get('adapter_lr') is None) != (args.adapter_lr is None):
            raise ValueError('Resume optimizer groups differ; keep --adapter-lr consistent')
        accelerator.load_state(str(args.resume))
        start_stage, start_step = progress['stage'], progress['step']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = accelerator.unwrap_model(model)

    ocr_teacher = FrozenOCRLoss(args.ocr_model, accelerator.device) if args.ocr_model else None

    def forward_losses(batch):
        losses, predictions = model(batch)
        if ocr_teacher is not None:
            with accelerator.autocast():
                ocr_loss = ocr_teacher(raw, predictions, batch['specials'],
                                      batch['gen_text'], batch.get('is_historical'))
            losses['ocr_loss'] = ocr_loss
            losses['loss'] = losses['loss'] + args.ocr_loss_weight * ocr_loss
        return losses

    global_start = sum(s.steps for s in stages[:start_stage]) + start_step
    writer = SummaryWriter(str(args.output_dir / 'tensorboard'),
                           purge_step=global_start + 1 if args.resume else None) if SummaryWriter and accelerator.is_main_process else None

    def evaluate(stage, val_loader, global_step):
        training = model.training
        model.eval()
        totals = torch.zeros(len(METRICS) + 1, device=accelerator.device)
        with isolated_rng(args.seed, accelerator.device), torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= args.eval_batches:
                    break
                inputs = raw.get_model_inputs(batch['style_img'], batch['gen_img'], batch['style_img_width'],
                    batch['gen_img_width'], 2 * stage.max_width + 16)
                losses = forward_losses({**batch, **inputs})
                count = len(batch['style_text'])
                totals += torch.stack([losses[k].detach() * count for k in METRICS] + [totals.new_tensor(count)])
        totals = accelerator.reduce(totals, reduction='sum')
        for i, key in enumerate(METRICS):
            if writer:
                writer.add_scalar(f'validation/{key}', (totals[i]/totals[-1].clamp_min(1)).item(), global_step)
        accelerator.print(f'{stage.name}: validation loss={(totals[0]/totals[-1].clamp_min(1)).item():.5f}')
        model.train(training)

    sample_recognizer = sample_alt_vae = None
    if accelerator.is_main_process and not args.no_samples:
        if args.sample_ocr:
            from .evaluate import Recognizer
            sample_recognizer = Recognizer(args.sample_ocr, 'cpu', 4, 256, args.output_dir / 'sample-ocr-cache.json')
        if args.sample_alt_vae:
            from diffusers import AutoencoderKL
            sample_alt_vae = AutoencoderKL.from_pretrained(args.sample_alt_vae).to(accelerator.device).eval()

    def save(stage_index, step):
        path = args.output_dir / f'stage-{stage_index}-step-{step}'
        accelerator.save_state(str(path))
        if accelerator.is_main_process:
            import json
            (path / 'progress.json').write_text(json.dumps({'input_pipeline_version': 3, 'replay_sampling_version': REPLAY_SAMPLING_VERSION if shards else None, 'data_epoch': epoch, 'stage': stage_index, 'step': step, 'height': args.height,
                'stages': [asdict(s) for s in stages], 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2))
        accelerator.wait_for_everyone()
        global_step = sum(s.steps for s in stages[:stage_index]) + step
        evaluate(stages[stage_index], val_loader, global_step)
        if accelerator.is_main_process and styles:
            preview_styles = (select_validation_styles(preview_data, args.sample_count, args.seed,
                args.sample_max_width or args.max_width, args.sample_fixed_count,
                (step - 1) // args.save_every,
                stratify_lengths=args.curriculum == 'three-stage') if paired_samples else styles)
            for cfg in args.sample_cfg_scales or [args.sample_cfg_scale]:
                for seed in args.sample_seeds or [args.seed]:
                    folder = path / 'samples'
                    if args.sample_cfg_scales or args.sample_seeds:
                        folder /= f'cfg-{cfg:g}-seed-{seed}'
                    metrics = checkpoint_samples(raw, preview_styles, folder, args.sample_text,
                        args.max_width, cfg, seed, writer, global_step, args.sample_max_width,
                        teacher_forced=args.sample_teacher_forced, alt_vae=sample_alt_vae,
                        recognizer=sample_recognizer, independent_pairs=bool(args.sample_cfg_scales or args.sample_seeds))
                    accelerator.print(f'Checkpoint previews: {folder}; {metrics}')
        if writer:
            writer.flush()
        accelerator.wait_for_everyone()

    for stage_index, stage in enumerate(stages):
        if stage_index < start_stage or stage.steps == 0:
            continue
        first_step = start_step if stage_index == start_stage else 0
        if first_step >= stage.steps:
            continue
        train = LinePairs(args.manifest, 'train', stage, args.curriculum, args.pair_by, args.use, args.height)
        val = LinePairs(args.manifest, 'val', stage, args.curriculum, args.pair_by, args.use, args.height)
        if paired_samples and not args.no_samples:
            preview_data = (LinePairs(args.manifest, args.sample_split, stages[-1],
                args.curriculum, args.pair_by, args.use, args.height)
                if args.sample_split == 'test' or args.curriculum == 'three-stage' else val)
            styles = select_validation_styles(preview_data, args.sample_count, args.seed,
                args.sample_max_width or args.max_width, args.sample_fixed_count,
                stratify_lengths=args.curriculum == 'three-stage')
        training_data = ReplayMix(train, shards, args.historical_probability, args.seed) if shards else train
        loader = accelerator.prepare(DataLoader(training_data, batch_size=args.batch_size, shuffle=not bool(shards),
            num_workers=args.workers, collate_fn=collate_pairs))
        val_loader = accelerator.prepare(DataLoader(val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_pairs))
        accelerator.print(f'{stage.name}: {len(train)} train targets, {len(val)} val targets; {stage}')
        if shards:
            accelerator.print(f'Replay: {len(shards)} shards; historical fraction={args.historical_probability}; '
                              f'{len(train)} historical targets per shuffled mixed pass ({len(training_data)} total pairs)')
        raw.teacher_noise = stage.teacher_noise
        model.train()
        step = first_step
        # Resume intentionally starts a fresh traversal. Advance the stream seed
        # instead of replaying the exact same initial historical/replay targets.
        epoch = (progress.get('data_epoch', first_step - 1) + 1
                 if args.resume and stage_index == start_stage else 0)
        train_totals = torch.zeros(len(METRICS) + 1, device=accelerator.device)
        # Each stage counts optimizer updates, including proper handling of a final partial accumulation.
        while step < stage.steps:
            if shards:
                training_data.set_epoch(epoch)
            if hasattr(loader, 'set_epoch'):
                loader.set_epoch(epoch)
            for batch in loader:
                with accelerator.accumulate(model):
                    texts = [text if torch.rand(()).item() >= stage.style_dropout else '' for text in batch['style_text']]
                    # The budget includes BOTH images and both visual delimiters.
                    inputs = raw.get_model_inputs(batch['style_img'], batch['gen_img'], batch['style_img_width'],
                        batch['gen_img_width'], 2 * stage.max_width + 16)
                    losses = forward_losses({**batch, **inputs, 'style_text': texts})
                    count = len(batch['style_text'])
                    train_totals += torch.stack([losses[k].detach() * count for k in METRICS] + [train_totals.new_tensor(count)])
                    accelerator.backward(losses['loss'])
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    step += 1
                    if step % args.log_every == 0 or step == stage.steps:
                        values = accelerator.reduce(train_totals, reduction='sum')
                        global_step = sum(s.steps for s in stages[:stage_index]) + step
                        if writer:
                            for i, key in enumerate(METRICS):
                                writer.add_scalar(f'train/{key}', (values[i]/values[-1].clamp_min(1)).item(), global_step)
                            writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
                            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                            for group in optimizer.param_groups:
                                writer.add_scalar(f"train/lr_{group.get('name', 'group')}", group['lr'], global_step)
                            writer.add_scalar('train/stage', stage_index, global_step)
                            writer.flush()
                        train_totals.zero_()
                    if step % args.log_every == 0:
                        accelerator.print(f'{stage.name} {step}/{stage.steps}: loss={losses["loss"].item():.5f}')
                    if step % args.save_every == 0 or step == stage.steps:
                        save(stage_index, step)
                    if step == stage.steps:
                        break
            epoch += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save({'input_pipeline_version': 3, 'model': raw.state_dict(), 'height': args.height, 'max_width': args.max_width,
                    't5_config': args.t5_config, 'vae': args.vae, 'head_config': raw.head_config}, args.output_dir / 'model.pth')
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
