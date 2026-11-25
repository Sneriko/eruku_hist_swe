# Historical Swedish checkpoint evaluation

`historical_swedish/evaluate.py` evaluates the images already saved by training.
It accepts either a single checkpoint directory or its parent training run.
It does not load the 8+ GB generator/optimizer checkpoints or regenerate samples.

```bash
.venv/bin/python -m pip install -r requirements-evaluation.txt

# Fast scan of training metadata, validation losses and generation failures:
.venv/bin/python -m historical_swedish.evaluate \
  output/historical-swedish-clean-vae

# Swedish recognition comparison across all checkpoints, same 10 fixed pairs:
.venv/bin/python -m historical_swedish.evaluate \
  output/historical-swedish-clean-vae --ocr --device cuda:0 \
  --output output/historical-swedish-clean-vae/analysis-ocr

# All paper metrics for the specific requested checkpoint, using all 20 previews:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -m historical_swedish.evaluate \
  output/historical-swedish-clean-vae/stage-1-step-5000 \
  --group all --ocr --visual HWD FID BFID KID --device cuda:0 \
  --output output/historical-swedish-clean-vae/analysis-final

# Compare candidate checkpoints on exactly the same fixed pairs:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -m historical_swedish.evaluate \
  output/historical-swedish-clean-vae --steps 1250 4550 4900 5000 \
  --visual HWD FID BFID KID --device cuda:0 \
  --output output/historical-swedish-clean-vae/analysis-visual
```

Outputs: `summary.csv`, `summary.json`, `ranking.json`, `predictions.json` and
`settings.json`. Recognition predictions are cached by image contents and model,
processor, tokenizer and decoding settings. Interrupted runs can reuse this cache.
The recognizer defaults to `Riksarkivet/trocr-large-handwritten-hist-swe-3-char`;
`--ocr-model` also accepts a local model directory. This machine already has its
weights cached. GPU 0 was used for this evaluation.

## What the metrics mean

The [Emuru paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Pippi_Zero-Shot_Styled_Text_Image_Generation_but_Make_It_Autoregressive_CVPR_2025_paper.pdf)
uses **FID, BFID, KID, HWD and absolute ΔCER**. The
[Eruku paper](https://arxiv.org/abs/2510.23240) uses **HWD, absolute ΔCER, FID and BFID**.
Lower values are preferable for these metrics.

| Metric | What it asks | Interpretation and limitations |
| --- | --- | --- |
| CER | Does an independent recognizer read the requested text? | Character insertions + deletions + substitutions, divided by reference characters. 0.10 means 10 errors per 100 reference characters. Can exceed 1 with many insertions. Reported separately for real and generated images. |
| Absolute ΔCER | Does generated handwriting have similar readability to the real handwriting? | `abs(CER_generated - CER_real)`. Matching a difficult reference can produce a small value without good readability. Read it alongside both CER values and style scores. |
| HWD | Does the writing look like the reference style? | Euclidean distance between mean features from a font-trained VGG network, averaged across style groups. These runs use page IDs as style proxies; a page is not a verified writer identity. |
| FID | Do real and generated image distributions look alike? | Compares means and covariances of Inception features, using a Gaussian approximation. Sensitive to image appearance and sample count. It is not a transcription check. |
| BFID | Do the ink shapes look alike after removing grayscale/background variation? | FID after Otsu binarization of both image sets. Reduces background influence, but can hide meaningful grayscale defects. |
| KID | Do the feature distributions match without a Gaussian assumption? | Polynomial-kernel maximum mean discrepancy in Inception feature space. The unbiased finite-sample estimate may be slightly negative. This script records raw KID; Emuru tables multiply it by 1,000. |
| WER (extra) | Are entire words correct? | Word edits divided by reference words. A single wrong letter makes a word wrong. |
| EOG rate (extra) | Did the generator emit its stop token? | A stop is not proof of correct text; inspect CER and blank-image rate too. |

Visual metrics use the [authors' HWD implementation](https://github.com/aimagelab/HWD),
pinned in `requirements-evaluation.txt`: height 32, its default initial-square crop
for FID/BFID/KID, and full-line handwriting features for HWD. The script includes
a narrow adapter for SciPy's removed `sqrtm(disp=...)` argument. It does not change
the metric formula. OCR uses the model's configured 192 × 1024 processor input
and ViT positional interpolation. ByT5 decoding disables artificial spaces between
added Swedish tokens. Text normalization is Unicode NFC only; spelling, case,
punctuation and whitespace are retained. CER/WER are corpus totals, not averages
of per-line rates. ΔCER is the absolute difference of the two corpus CERs.

This Swedish model and dataset differ from the papers' English recognizer and
benchmarks. The resulting values are useful within this run, not as direct
comparisons to their published tables.

## Choosing checkpoints and interpreting passes

Use the default fixed group for checkpoint comparisons. The other ten previews
rotate between checkpoints, so `--group all` is useful to inspect one checkpoint
but does not provide an identical comparison set across checkpoints. The script
checks target identity, text and reference identity before ranking fixed groups.
A missing/error generation receives an empty OCR prediction, so its characters
count as deletions; near-blank and non-stopping images are also kept in CER.
Visual metrics exclude missing/error images and report the retained count.

The fixed subset has only 10 lines / 275 characters. Selecting among 100
checkpoints on this tiny subset risks selection bias. Treat rankings as candidate
screening; confirm candidates on a larger fixed validation set and reserve test
data for final evaluation. FID/BFID/KID from 10 or 20 images are particularly
unstable. Validation loss uses teacher forcing and only the configured 20 batches;
it measures a different task from autonomous generation.

For this run, the long stage has 94,092 eligible historical targets. Replay
sampling version 2 visits each once per mixed traversal, adding font pairs to
reach `ceil(94092 / 0.8) = 117615` total pairs. With batch size 8, accumulation 32
and one training process, a traversal is about 14,702 microbatches / 460 optimizer
updates, including a final partial accumulation. Thus step 1,250 is approximately
2.72 passes, 4,550 is 9.89, 4,900 is 10.65, and 5,000 is 10.87 passes. These are
passes over **eligible historical target lines**, not all manifest rows, every
possible reference–target combination, or the replay corpus. Reference lines are
sampled within each page and can be reused.

`progress.json:data_epoch` is a zero-based traversal index: 10 means the eleventh
traversal was in progress. Pass estimates assume unchanged loader configuration
and no partial-traversal resumes. Use `--world-size` if training used more than one
process. These counts describe this adaptation run; they do not include the
original Eruku checkpoint's pretraining or the separate VAE training.

After a completed sweep, regenerate the combined report and PNG/SVG curves with:

```bash
.venv/bin/python scripts/report_historical_evaluation.py \
  output/historical-swedish-clean-vae
```

The measured report is in
[`output/historical-swedish-clean-vae/analysis-report/report.md`](output/historical-swedish-clean-vae/analysis-report/report.md).
The current provisional choice is step 4,550: 17.45% CER on the fixed previews,
tied with step 4,700, with better WER (30.61% vs 36.73%). Step 1,250 is almost tied
at 17.82% CER with fewer passes and better HWD among the visual shortlist.
The paired bootstrap interval for CER(4550) minus CER(5000) crosses zero, so the
small preview set does not establish a statistically clear winner.
