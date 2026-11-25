# Optional autoregressive diffusion head

The default `--latent-head linear` retains the existing model and MSE objective.
`--latent-head diffusion` retains T5, the frozen VAE, latent input projection,
and SOG/IMG/EOG classifier, and adds a conditional MLP to generate each VAE
column. The old `t5_to_vae` weights remain in checkpoints but are frozen and
unused for diffusion image predictions. At 128px with the current one-channel
VAE, each column is a 16-dimensional vector and advances eight image pixels.

The head uses cosine variance-preserving diffusion with velocity prediction
and deterministic DDIM updates from random initial noise. Classifier-free
guidance combines conditional/unconditional denoiser predictions at every
sampling step. EOG still comes from the existing classifier. This is a local
column generator, not whole-image denoising, and cannot guarantee correct text,
no repetition, or correct stopping.

## Run the pilot

From the repository, run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_diffusion_experiment.sh
```

The script uses the existing step-3750 checkpoint, production-century-prose-128
manifest, trained 128px VAE, and 50/50 historical/replay mixture. It preserves
history noise (25% probability, 0.05 relative std), text dropout, and preview
CFG. It runs two separate training processes:

1. **250 optimizer updates**, LR `1e-3`, denoiser only; all original weights frozen.
2. **2,000 optimizer updates**, LR `2e-5`, joint training of the denoiser and
   original trainable components; VAE and old linear output head stay frozen.

These are initial experimental settings, not established optimal values. Each
phase starts a new optimizer. `--init-checkpoint` transfers weights;
`--resume` restores an interrupted run with the same architecture/freezing mode.
The second phase deliberately uses initialization rather than resume.

The script can run one phase at a time:

```bash
bash scripts/train_diffusion_experiment.sh warmup
bash scripts/train_diffusion_experiment.sh joint
```

Outputs go to `output/eruku-century-prose-128-diffusion-pilot/{warmup,joint}`.
Override `EXPERIMENT_BASE` or `EXPERIMENT_OUTPUT` to choose different paths.
`BATCH_SIZE` and `ACCUMULATION` default to 16 and 64. For less memory, use
`BATCH_SIZE=8 ACCUMULATION=128` to preserve the effective batch on one GPU.
Do not start a second process targeting a directory already being trained.

## Generate from a saved checkpoint

Both final `model.pth` files and intermediate Accelerate checkpoint directories
are supported; architecture settings are loaded automatically:

```bash
.venv/bin/python -m historical_swedish.generate \
  --checkpoint output/eruku-century-prose-128-diffusion-pilot/joint/stage-1-step-250 \
  --style-image output/eruku-century-prose-128-vae-replay50-noise25-std05-from3250-gpu0/stage-1-step-3750/samples/45-style.png \
  --style-text 'att wittnet skulle låta bli sorga, och' \
  --text 'går alt wara sant, hwad detta sednare' \
  --output output/diffusion-example.png \
  --max-width 4096 --cfg-scale 1.375 --diffusion-steps 20 --seed 24
```

Preview generation uses the same diffusion path. The existing
`scripts/sweep_cfg_samples.py` also supports diffusion checkpoints. Sampling
cost increases: each column needs the configured number of small denoiser
passes, in addition to the autoregressive backbone. Keep seeds fixed for
comparisons; diffusion adds a new source of sampling randomness.

## Training and comparison details

Training predicts velocity from noisy **clean target columns** and causal T5
contexts. Only IMG positions contribute to diffusion loss (including style
positions, matching the original MSE coverage); SOG/EOG/padding do not.
History noise is separate and never changes the clean denoising target.
The total objective is special-token CE plus diffusion velocity MSE.
`diffusion_loss` is logged separately. In diffusion mode, `mse_loss` is a
single-timestep clean-column reconstruction diagnostic; it is not the original
linear regression metric and is not added to the objective. OCR auxiliary
supervision is disabled for this initial diffusion experiment.

`--diffusion-latent-scale` is a fixed positive divisor used only inside the
head, reversed before returning columns. Default 1.0 retains the repository's
raw VAE latent convention; the VAE config's `scaling_factor` is not silently
applied. Do not change scale between training and inference. Width, depth,
scale, head type, and sampling steps are saved in checkpoint metadata. Resume
checks these settings; inference can override sampling steps explicitly.

Compare held-out generated images and OCR/CER, repetition frequency, near-blank
rate, EOG rate, and output width at matching prompts/styles. The existing
preview success rate checks EOG and blankness, not transcription accuracy.
Training loss values from linear and diffusion runs are not directly comparable.
A matched linear continuation from the same source checkpoint is the useful
control for judging whether the new head helps.
