# Historical Swedish Eruku

This adds PAGE XML/image preparation, 64-pixel generation, a default maximum
**target** width of **2048 pixels**, and two-stage curriculum training.
The existing IAM/WebDataset scripts remain legacy entrypoints; use the modules
below for historical Swedish.

## Research and choices

Read before implementation:

- [Emuru, CVPR 2025](https://arxiv.org/html/2503.17074v1), §§3.3 and 4:
  4–7 words, 70,000 updates; then 1–32 words, 250 updates. Original widths
  768 and 2048 at height 64. First stage uses noisy teacher forcing with
  probability 0.1; second stage has no teacher noise.
- [Eruku, WACV 2026](https://arxiv.org/html/2510.23240v1), §§3–4:
  first stage has 2–3 style words and 2–3 target words, 65,000 updates;
  second stage has 1–8 style words and 1–32 target words, 5,000 updates.
  Style-text dropout is 0 in stage one and 0.1 in stage two. Text-unconditional
  training remains enabled in both stages. The paper chooses 0.1 style dropout
  in its ablation and text CFG 1.25 at inference.

For this historical-Swedish adaptation, **style images must contain at least
three words** in training and validation. `--curriculum eruku` is the default:
style ranges are **3–8 words in both stages**, with targets of 2–3 words in the
short stage and 1–32 words in the long stage. This deliberately changes the
paper's style ranges based on observed quality with longer style references.

`--curriculum emuru` applies Emuru's
**combined** word-count ranges to Eruku style/target pairs. It retains Eruku's
SOG/EOG and CE+MSE objectives; it is an Emuru-inspired schedule, not a reproduction
of Emuru's architecture or loss. Its style ranges are 3–6 words in the short
stage and 3–31 words in the long stage, while combined style+target ranges remain
4–7 and 1–32 words respectively (the three-word style minimum makes the smallest
combined sample four words). Width caps are 768/2048 pixels at native height 64.
For Eruku these are per-image width caps; Emuru mode also limits the combined
pair to its stage width. The Transformer sequence budget includes both images
plus two delimiter tokens. Inputs are dynamically padded to multiples of 8.
Over-wide data is rejected, never clipped with its transcription left intact.

The VAE stays frozen, as in Eruku. The default native height is **64 pixels**,
matching the published checkpoint’s eight-value vertical latent slices and image
adapters. This preserves all compatible pretrained image-adapter weights.
For TrOCR input at 192px, resize the final generated image proportionally with
`--output-height 192`; this does not add native high-resolution stroke detail.

## Install

Install a compatible PyTorch/torchvision build for your hardware, then:

```bash
pip install -r requirements-historical.txt
```

Model/tokenizer assets are downloaded from Hugging Face on the first run.
`--t5-config` selects a T5 architecture configuration; it does **not** load
pretrained T5 language weights. Without `--init-checkpoint` or `--resume`, the
Transformer is initialized randomly, matching the original implementation.
Historical training does not require font rendering, NLTK, IAM, Karaoke, wandb,
or the optional Origami OCR checkpoint.

## Prepare PAGE XML and images

The extractor, background normalization and group splitter are reused from
`/home/marerle/projects/emuru-hist-swe/historical_swedish`, with height/width
changes and optional word-span extraction added here. The source repository's
MIT license applies to reused code; see `LICENSE.emuru-hist-swe`.

```bash
python -m historical_swedish.extract_page_lines \
  --root /data/swedish-pages /data/another-dataset --output /data/swedish-eruku \
  --height 64 --max-width 2048 --word-spans --strict

python -m historical_swedish.build_splits \
  --input /data/swedish-eruku/lines.jsonl \
  --output /data/swedish-eruku/split.jsonl \
  --group-by document_id --val-fraction 0.05 --test-fraction 0.10
```

Pass one or several dataset paths after `--root` (alias `--roots`), or repeat
the option: `--root /data/a --root /data/b`. All datasets are written to one
manifest and crop directory. Duplicate and overlapping roots are deduplicated.
Each row records `dataset_root` and a `dataset_id` derived from its resolved root
path. With multiple roots, page/document/volume/writer/archive IDs, including
CSV-supplied IDs, are prefixed with this dataset ID to prevent accidental merges.
Keep root paths stable across extraction runs to retain these identities.
For metadata specific to a dataset, use absolute XML paths in the CSV to avoid
ambiguous relative filenames shared by datasets.

PAGE namespaces are supported. Images may sit next to XML, beside a `page/` folder, or in a sibling
`image/` folder (e.g. `dataset/page/p.xml` with `dataset/image/p.png`).
PAGE `imageFilename` is consulted, with matching XML stems as a fallback. Text uses NFC and whitespace
normalization; historical spelling, long s (`ſ`) and Swedish characters survive.
Default training uses conservative grayscale illumination correction, without
binarization. `--use raw` selects the original masked crops instead.

Optional `--metadata-csv metadata.csv` on extraction accepts `xml_path` or
`page_id` plus `document_id`, `volume_id`, `writer_id`, `pseudo_writer_id`,
`archive_id`, and `year`. Supply real document/volume identities when possible.
The fallback document group is the enclosing directory; a flat directory becomes
one document and therefore one split. Hash splits do not guarantee nonempty
train/validation/test sets for small collections. Inspect reported split counts.
Training requires train and validation pairs for every enabled stage.

Pairs use distinct lines from the same page and split. With reliable writer
metadata, `--pair-by pseudo_writer_id` allows pairing across pages of one writer.
Choose split groups that keep these writers together if evaluating unseen writers.

Short-stage filtering uses complete aligned lines. `--word-spans` also extracts
2–3-word windows when PAGE contains Word coordinates and word transcriptions
agree with line text. These use the parent line's vertical extent to preserve
scale. Windows follow PAGE word order and span the selected word bounding boxes;
inspect crops for slanted/overlapping handwriting. A span and another span or
full line from the same original line are never paired. With line-only PAGE,
short stage needs real 2–3-word target lines and distinct 3–8-word style lines
in the same group; text is never cut at guessed pixel positions. The extracted
2–3-word spans can serve as targets, and cannot serve as style references.
If none qualify, training fails with a diagnostic. For adaptation directly on
long lines, explicitly use `--short-steps 0`.

## Training audit and paper comparison

The current generator is not yet a validated historical-Swedish adaptation.
See the saved [data and paper audit](output/diagnostics/training-audit/DATA_AND_PAPER_AUDIT.md)
and [experimental results](output/diagnostics/training-audit/REPORT.md).
Early acceptance means recognizable handwriting structure and natural EOG
stopping. Accurate transcription is a later goal. Matched validation generation
revealed regressions after fine-tuning despite improving teacher-forced loss.

**Correction to the earlier padding diagnosis:** Eruku §5 explicitly trains
repeated visual EOG labels in batch padding. They are intentional supervision,
not bogus labels. The experimental version 2 pipeline masked padding with -100;
that change did not establish a quality fix and has been reverted. Version 3
restores repeated EOG while latent MSE continues to exclude non-image positions.
Boundary metrics count only the first, real EOG so padding cannot inflate the
reported stopping accuracy.

VAE inputs are still encoded at each image's own rounded-to-eight width before
latent padding. The VAE's spatial normalization otherwise changes a line's
representation with unrelated batch neighbours. This change has a batch-invariance
regression test, but is not by itself a demonstrated solution to generation drift.
Version 3 checkpoints reject older pipelines on resume; use the original native64
checkpoint and a new output directory for controlled comparisons. Existing crops
and manifests remain usable.

The paper's second stage uses about 10M synthetic pairs, 5000 iterations, AdamW,
learning rate1e-4, weight decay1e-2 and a virtual batch 256. This is not a validated
recipe for adapting to 94,092 eligible historical targets. At 90% historical
sampling, 5000 updates draw approximately 1.15M historical targets, or 12.2 dataset-size
equivalents in the earlier replacement-sampling experiments. Fifty updates draw 11,520 historical targets
(0.122 equivalents). The 90/10 mixture and learning rate 2e-5 are adaptation choices,
not values established by the paper. The final published Eruku does not use an
auxiliary OCR loss. See [Eruku §§4–5](https://arxiv.org/html/2510.23240v1).

Use `--sample-validation-pairs` to generate fixed held-out target transcripts and
save `NN-style.png`, `NN-target.png`, and `NN-generated.png`. Target pixels are
saved for comparison only and never enter generation. Without this option,
`--sample-style-dir` and `--sample-text` provide the fixed-prompt probe. Keep the
same seed and widths across comparisons. Preview `success_rate` requires nonblank
output and natural EOG; it does not classify handwriting structure or accuracy.

TensorBoard includes teacher-forced `boundary_ce_loss`, `sog_accuracy`,
`eog_accuracy`, and the pre-clipping `train/grad_norm`. Free-generation
`samples/eog_rate` is the relevant check for natural stopping.

Optional `--freeze-decoder`, `--freeze-decoder-ffn`, and
`--ocr-model Riksarkivet/trocr-large-handwritten-hist-swe-3-char
--ocr-loss-weight 0.1` are experimental controls, not recommended solutions.
The optional OCR critic is frozen, supervises only predicted historical target
regions, and excludes replay/reference pixels. Its own OCR score is not an
independent quality evaluation. Neither freezing experiment has reliably fixed
the observed collapse. Do not launch a 5000-update run assuming these controls
or the VAE-padding change solve it; inspect a bounded pilot first.

## Train

```bash
CUDA_LAUNCH_BLOCKING=0 accelerate launch --num_processes 1 -m historical_swedish.train \
  --manifest /data/swedish-eruku/split.jsonl \
  --output-dir output/historical-swedish \
  --curriculum eruku --height 64 --max-width 2048 \
  --batch-size 2 --accumulation 128 --precision bf16
```

Stage durations count **optimizer updates**. Defaults follow the papers; choose
shorter durations for a small real corpus or a smoke run, e.g.
`--short-steps 2 --long-steps 2 --accumulation 1 --precision no`.
Physical batch size is configurable and defaults to 2 for memory use.
Effective batch = batch size × accumulation × process count; defaults give 256
on one GPU. Adjust accumulation for multiple GPUs. The configurable probability
of replacing all text embeddings with the learned unconditional embedding is
`--text-dropout 0.05` in both stages. Emuru mode adds unit Gaussian teacher noise
to a random 0.1 fraction of first-stage samples.

To initialize from the published Hugging Face Eruku checkpoint:

```bash
# Add to the training command:
--init-checkpoint blowing-up-groundhogs/eruku
```

The model ID downloads `000073688.pth` from the model repository using the
Hugging Face cache; subsequent runs reuse the cached file. No remote Python code
is executed. Use `--init-checkpoint-revision COMMIT` to pin a version or
`--init-checkpoint-filename NAME.pth` for another compatible weight file.
A local `.pth` file path also remains supported. Keep `--t5-config` matched to
the pretrained architecture (the default `google-t5/t5-large` matches this model).

Checkpoint initialization is strict about image-adapter dimensions by default.
Use the original Eruku checkpoint for 64px training, not the experimental 192px
weights. Legacy height transfer remains an explicit experimental option via
`--adapter-init interpolate` or `--adapter-init random`; neither is used by the
64px workflow. Compatible T5 and special-token weights are loaded, and optional
upstream OCR weights are unused. Initialization starts a fresh optimizer; it is
not an exact training resume.

### Reuse existing crops

You do **not** need to repeat PAGE XML extraction or split generation when moving
from 192px training to native 64px. Keep your existing `split.jsonl` and image
files. The loader checks each image against its original manifest dimensions,
then resizes it proportionally to the requested training height with Lanczos.
Width filtering and pairing use the resized geometry: a 6000×192 crop becomes
2000×64 and fits the default 2048px limit. Images and transcriptions are never
cropped, and the files, metadata, and train/validation/test membership are unchanged.
Synthetic replay and preview style references are also resized to 64px.

Re-extract only if you need different crop boundaries or word spans, or original
page crops at a new resolution without the extra resampling step. New extraction
now defaults to 64px and maximum width 2048px.

Start a **new run** from `blowing-up-groundhogs/eruku` using a new output directory.
The old 192px optimizer checkpoints cannot be resumed as 64px training. No adapter
warm-up or height interpolation is needed for the original 64px checkpoint.

### Optional synthetic replay

For adaptation from an existing checkpoint, mix historical data with locally
downloaded **blowing-up-groundhogs/font-square-pairs-ft** paired WebDataset shards.
Download a subset with the resumable helper (16 shards by default):

```bash
# Preview selected files and total size first; this downloads no dataset files.
python scripts/download_fontsquare_replay.py --num-shards 16 --dry-run

# Download to /mnt/scratch/emuru_eruku_data/data/font-square-pairs-ft.
python scripts/download_fontsquare_replay.py --num-shards 16
```

Use `--output-dir /your/path` to change the destination, `--start-shard 16` to
select a later subset, or `--all` for the full dataset. Repeating the same command
reuses completed files and resumes interrupted downloads. The helper resolves
actual shard names from Hugging Face and pins a dataset commit during each run;
use `--revision COMMIT` to keep that same version across runs. It writes
`shards.txt` after successful completion, listing exactly the selected files
(the list is replaced on a subsequent successful download selection).

Use that list with training:

```bash
--replay-train-pattern /mnt/scratch/emuru_eruku_data/data/font-square-pairs-ft/shards.txt
```

Next controlled diagnostic: physical batch2, matching the paper’s long stage
(unvalidated for this historical corpus; compare against the saved batch8 pilots):

```bash
CUDA_LAUNCH_BLOCKING=0 accelerate launch --num_processes 1 -m historical_swedish.train \
  --manifest /mnt/scratch/emuru_eruku_data/swedish_historical_eruku/split.jsonl \
  --output-dir output/historical-swedish-replay-64-audit-pilot \
  --init-checkpoint blowing-up-groundhogs/eruku \
  --replay-train-pattern '/mnt/scratch/emuru_eruku_data/data/font-square-pairs-ft/*.tar' \
  --historical-probability 0.9 \
  --short-steps 0 --long-steps 50 \
  --height 64 --max-width 2048 \
  --batch-size 2 --accumulation 128 --workers 4 \
  --precision bf16 --lr 2e-5 --log-every 1 --save-every 50 \
  --sample-max-width 1024 --sample-validation-pairs
```

This example skips short-stage training for direct adaptation to long lines.
This is a baseline diagnostic pilot; inspect generation before extending it.
The optional experimental controls are described above.
Without `--replay-train-pattern`, training remains historical-only. With replay,
`--historical-probability` defaults to 0.9: approximately 90% historical and 10%
synthetic **samples**, mixed within batches. It must be in (0, 1]; 1 disables
synthetic sampling. Validation always uses historical validation pairs only.

Supply a quoted local glob, a single `.tar`, or a `.txt` list containing one
shard path per line (relative paths are resolved against the list's directory).
Download only the shards you want to use; no automatic download is performed.
These must be paired shards with `style.rgb.png`, `gen.rgb.png`, `style.bw.png`,
`gen.bw.png` and JSON `style_text`/`gen_text`, rather than single-image Emuru
`font-square-v2` shards. Default normalized mode uses the supplied `bw` images;
`--use raw` uses RGB. Synthetic images are resized proportionally to the training
height with Lanczos interpolation. Upscaling does not add native stroke detail.
Pairs are filtered by stage word counts (including the three-word style minimum)
and resized widths; images/text are never truncated. A full shard pass without
eligible pairs fails with a diagnostic. Supply more shards if filtering removes
all pairs, especially for short-stage training.

Shards stream from disk and repeat as needed. A mixed pass now visits every
eligible historical target once in shuffled order, without replacement. With
`N` historical targets and historical fraction `p`, the pass contains
`ceil(N / p)` total pairs: all `N` historical targets plus replay. The source order
is shuffled, and workers take disjoint parts of the same global plan. Historical
references are still chosen randomly from eligible distinct lines. After the
pass finishes, historical targets reshuffle for the next pass.

Optimizer-update stage durations are unchanged. Checkpoints record
`replay_sampling_version: 2`. Older replacement-sampling checkpoints cannot be
silently resumed with the new sampler. Resume starts a new shuffled pass rather
than restoring an exact position inside the interrupted pass, so the no-repeat
coverage guarantee applies to completed, uninterrupted passes. Replay settings
and raw/normalized mode must match on resume. For a historical-only finishing
phase, start a new run using `--init-checkpoint .../model.pth` and omit replay.

Accelerator checkpoints include optimizer/RNG state, stage and update position:

```bash
# Add to the same training command with matching curriculum and height:
--resume output/historical-swedish/stage-0-step-500
```

Resume restores optimizer state and the remaining curriculum updates, but starts
a new data-loader traversal; it does not reproduce the exact next minibatch.
Keep all training/data arguments consistent on resume. Validation reports
teacher-forced loss without conditioning dropout at each completed stage.
No test samples are used for training or validation. Final inference weights and
architecture metadata are saved in `output/historical-swedish/model.pth`.

## TensorBoard and checkpoint previews

TensorBoard logging and checkpoint previews are enabled by default. Install the
updated dependencies with `pip install -r requirements-historical.txt`, then run:

```bash
tensorboard --logdir output/historical-swedish-replay-64-audit-pilot/tensorboard --port 6006
```

Open http://localhost:6006 (forward port 6006 when training remotely).
Training logs sample-weighted total loss, special-token cross-entropy, latent MSE,
learning rate and curriculum stage every `--log-every 10` optimizer updates,
including the final update of each stage. Historical validation runs at **every
checkpoint**, using up to `--eval-batches 20` batches, and logs the same losses.
The TensorBoard horizontal axis counts cumulative optimizer updates across stages.
These losses are training diagnostics, not OCR character error rate.

At each `--save-every` checkpoint and each stage's final checkpoint, the main
process generates previews using these defaults (override on the training command):

```bash
--sample-style-dir /mnt/scratch/emuru_eruku_data/data/example_inference \
--sample-count 10 \
--sample-text 'Hej, jag heter Erik och jag tränar Eruku' \
--sample-cfg-scale 1.25
```

For five fixed and five fresh validation previews per checkpoint, use:

```bash
--sample-validation-pairs --sample-count 10 --sample-fixed-count 5
```

Preview slots00–04 keep both target and style reference unchanged. Slots05–09
rotate through the remaining eligible validation targets in disjoint blocks of
five, wrapping only when that pool is exhausted. Fixed targets are excluded from
the rotating pool. Selection is reproducible by seed, stage and checkpoint index,
including resume with the same save interval. Each JSON record includes
`preview_group` (`fixed` or `rotating`) and `target_key`. TensorBoard logs both
the aggregate rates and separate `samples/fixed/*` and `samples/rotating/*` rates.
The rotating group gives variety; use the fixed group for checkpoint comparisons.
Validation losses still use `--eval-batches`; these ten are generation previews.
Omitting `--sample-fixed-count` keeps all validation previews fixed.

Without `--sample-validation-pairs`, the style-directory mode behaves as follows:

Ten distinct styles are sampled once using `--seed` and reused for comparison
across checkpoints. At least ten are required; `--sample-count` can increase this.
Images are found recursively. Matching `.txt` files provide style transcriptions;
when present, styles with fewer than three words are excluded. References that
exceed `--max-width` after proportional resizing are excluded too. Missing or
insufficient eligible styles fail before training starts.

Each checkpoint's `samples/` directory contains numbered `*-style.png` references,
`*-generated.png` outputs and `samples.json` recording source paths, both texts,
generated widths and whether EOG was reached. References and outputs also appear
in TensorBoard's Images tab; success and EOG rates appear as scalar metrics.
Each generated image also records `pixel_min`, `pixel_max`, `pixel_std`,
`dark_fraction` (fraction below gray level 200), and `near_blank` (less than
0.1% of pixels below 200). TensorBoard logs `samples/near_blank_rate`; nearly
blank images are excluded from `samples/success_rate`. This is a conservative
blank-output check, not a readability or OCR score; dark stripes can still pass.
Standalone generation writes the same diagnostics and EOG status beside the PNG
in a `.json` file. Use `--seed 24` for repeatable standalone comparisons.

An early model can emit no image tokens: that attempt is recorded as a failure
in JSON and the success rate, with no fabricated generated PNG. Generation uses
`--sample-max-width` when supplied, otherwise the full `--max-width`, even during the short stage; hitting that limit
can yield incomplete text. Ten autoregressive generations can substantially
increase checkpoint time, particularly early in training.

Preview generation and validation preserve the training RNG states and restore
training mode afterward. Only the main process generates previews; other ranks
wait. Use `--no-samples` or `--no-tensorboard` to disable either feature independently.
Checkpoint resume appends to the existing TensorBoard directory and purges logged
steps after the resumed checkpoint. Keep preview settings and source files stable
across resumes for comparable images.

## Generate

```bash
python -m historical_swedish.generate \
  --checkpoint output/historical-swedish/model.pth \
  --style-image /data/example-line.png \
  --text 'Här följer en historisk svensk textrad.' \
  --output generated.png --max-width 2048 --cfg-scale 1.25
```

Optional `--style-text '…'` supplies its transcription. Output is grayscale,
64 pixels tall with dark ink and a light background. Only generated image
slices are decoded; reference and visual delimiter tokens are excluded.
Add `--output-height 192` to export a proportionally resized image for TrOCR.
The JSON sidecar records both native and exported dimensions. Generation and
`--max-width` limits are in native 64px coordinates; a 1000×64 result becomes
3000×192 on export.

Generation stops on EOG or the width/iteration budget and reports whether EOG
was reached. Hitting the budget can leave incomplete text; a width bound is not
a guarantee of text completeness.

## Verification

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests use tiny randomly initialized real T5/VAE models, without network or GPU:
PAGE XML and Swedish Unicode extraction, aligned spans/pairing, native-height
forward/backward at 64 and 192px, EOG-preserving sequence budgets, generation,
and curriculum training/checkpoint resume. Full-scale training and visual quality
need the actual dataset and suitable GPU resources.

## Raw-image 50/50 experiment (3000 optimizer updates)

Start the prepared launcher from this repository:

```bash
bash scripts/train_historical_raw_50_50.sh
```

This starts a fresh long-stage adaptation from original pretrained Eruku, using
existing **raw line crops**, RGB replay,50% historical data and50% replay. It
keeps height64, max-width2048, physical batch8, accumulation32, LR2e-5, and skips
the short stage as requested. It is not the paper's full two-stage curriculum.
No re-extraction is needed because the manifest already contains `raw_path`.

The3000 steps are **optimizer updates**, each normally accumulating32 minibatches
of8 pairs (effective256). Roughly768,000 total pair draws /384,000 historical
pairs give just over four shuffled historical passes with the current94,092
targets. A pass-ending partial accumulation can make the exact count slightly
smaller. Historical targets repeat only after the preceding pass completes.

Every50 updates the run saves a checkpoint, evaluates20 batches, and generates
five fixed plus five rotating validation pairs. Slots00–04 remain fixed and
slots05–09 change at each checkpoint. The output directory is
`output/historical-swedish-replay-64-raw-hist50`; the launcher refuses to reuse an
existing directory for a fresh run. It captures `training.log` and copies the
training source into `source/` for analysis. To choose a different GPU/output:

```bash
ERUKU_GPU=1 ERUKU_OUTPUT_DIR=output/historical-raw-hist50-v2 \
bash scripts/train_historical_raw_50_50.sh
```

TensorBoard:

```bash
.venv/bin/tensorboard --logdir output/historical-swedish-replay-64-raw-hist50/tensorboard --port 6006
```

## Optional target-history teacher noise

Add these training flags to enable a conservative experiment:

```bash
--target-noise-probability 0.1 --target-noise-std-ratio 0.05
```

The default probability is zero, so existing Eruku training behavior is unchanged.
For each selected training sequence, independent Gaussian noise is added only to
IMG latents between SOG and the first EOG. Style-reference latents, SOS/SOG/EOG
embeddings, and padding stay clean. The next-token MSE targets also stay clean.
This applies to both historical and replay examples. Evaluation and generation
never add this teacher-forcing noise.

Probability and magnitude are separate:0.1 selects10% of sequences, while0.05
sets the Gaussian standard deviation to5% of that sequence's clean target-latent
standard deviation. Scale is measured in fp32 over valid target slices and latent
features (population variance), excluding style and padding. A constant target
has zero measured scale. This is a relative magnitude, not absolute unit noise.
These values are experimental starting points, not a validated fix for collapse.

The target-only mode replaces legacy Emuru-style all-history noise when its
probability is positive. If disabled, any legacy curriculum noise retains its
existing behavior. The new flags are stored in checkpoint `progress.json` args.
Noise settings can intentionally change on resume; the trainer prints the change
and applies the requested settings from the next update. Use a different output
directory for a noise/control branch so both experiments remain available.

The raw50/50 launcher forwards additional training arguments, for example:

```bash
ERUKU_OUTPUT_DIR=output/historical-swedish-replay-64-raw-hist50-noise \
bash scripts/train_historical_raw_50_50.sh \
  --target-noise-probability 0.1 \
  --target-noise-std-ratio 0.05
```

This example starts a fresh3000-update long-stage run from pretrained Eruku, with
the same raw/replay settings and five fixed/five rotating previews. It does not
alter an already-running process. For a comparison from the same intermediate
weights and optimizer, use the normal `historical_swedish.train --resume ...`
command, keeping the previous replay, image mode, curriculum, batch and preview
settings and setting a new `--output-dir`; add the two noise flags to the treatment
branch only. Keep `--target-noise-probability 0` for the control. Resuming starts a
fresh data traversal as documented above.

## Test-set checkpoint gallery

Checkpoint previews now default to 50 fixed test-set pairs (`--sample-split test
--sample-count 50`). Every checkpoint's `samples/index.html` shows style, target,
and generated images with both transcriptions. Individual PNGs, `NN-text.txt`,
and `samples.json` are also saved. No-image generation failures are recorded
explicitly. Test targets never enter generation or loss evaluation; style
references come from distinct lines in the same test group. Preview eligibility
uses the long-stage word ranges in both training stages. At least 50 eligible
test targets within the preview width limit are required.

The same pairs and generation seed are reused for comparisons.
`--sample-fixed-count` enables a rotating subset. `--sample-split val` (or the
legacy `--sample-validation-pairs`) selects validation previews instead;
`--sample-split style` restores directory-based previews. `--no-samples` disables
previews. These settings take effect when starting/restarting training, not in
an already running process.
