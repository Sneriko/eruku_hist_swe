"""Summarize a completed evaluate.py OCR sweep and optional visual shortlists."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_swedish.evaluate import edit_distance, normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    rows = json.loads((run/'analysis-ocr/summary.json').read_text())
    predictions = json.loads((run/'analysis-ocr/predictions.json').read_text())
    out = run/'analysis-report'
    out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows.sort(key=lambda r:r['step'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot([r['step'] for r in rows], [r['cer_generated']*100 for r in rows], label='Generated CER')
    axes[0].axhline(rows[0]['cer_real']*100, color='gray', linestyle='--', label='Real reference CER')
    axes[0].set_ylabel('Character error rate (%)')
    axes[0].legend()
    axes[0].set_title('Historical Swedish: 10 fixed held-out lines, 275 characters')
    axes[1].plot([r['step'] for r in rows], [r['validation/loss'] for r in rows], color='tab:orange')
    axes[1].set_ylabel('Teacher-forced validation loss')
    axes[1].set_xlabel('Optimizer updates')
    for ax in axes:
        for step in (1250,4550,4700,4900):
            ax.axvline(step, linewidth=.7, alpha=.3, color='black')
        ax.grid(alpha=.2)
    for ext in ('png','svg'):
        fig.savefig(out/f'checkpoint-curves.{ext}', dpi=180)
    plt.close(fig)
    # Paired resampling keeps reference lines aligned. This is descriptive only;
    # it does not correct for choosing among 100 checkpoints on the same lines.
    groups = {}
    for row in predictions:
        groups.setdefault(row['checkpoint'], []).append(row)
    chosen = groups['stage-1-step-4550']; final = groups['stage-1-step-5000']
    assert [r['target_key'] for r in chosen] == [r['target_key'] for r in final]
    lengths = np.array([len(normalize(r['text'])) for r in chosen])
    diffs = np.array([edit_distance(normalize(a['text']),normalize(a['generated_ocr']))-
                      edit_distance(normalize(b['text']),normalize(b['generated_ocr'])) for a,b in zip(chosen,final)])
    index = np.random.default_rng(24).integers(0,len(chosen),size=(10000,len(chosen)))
    low,high = np.quantile(diffs[index].sum(1)/lengths[index].sum(1),[.025,.975])
    visual = {}
    for directory in ('analysis-visual','analysis-4700'):
        path=run/directory/'summary.json'
        if path.exists():
            visual.update({r['step']:r for r in json.loads(path.read_text())})
    lines = ['# Historical Swedish checkpoint analysis','',
        'Provisional choice: **stage-1-step-4550**, approximately **9.89 historical-data passes**. '
        'It ties step 4700 for lowest fixed-preview CER, has lower WER, and both stop on all ten examples. '
        'Step 1250 is a strong early candidate after only 2.72 passes.','',
        '| Step | Passes | Generated CER | ΔCER | WER | EOG | HWD | FID | BFID | KID × 1000 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for step in (1250,4550,4700,4900,5000):
        r=next(r for r in rows if r['step']==step);v=visual.get(step,{})
        vv=[f'{v[k]:.3f}' if k in v else 'pending' for k in ('hwd','fid','bfid')]
        vv.append(f'{v["kid"]*1000:.3f}' if 'kid' in v else 'pending')
        lines.append(f'| {step} | {r["estimated_passes"]:.2f} | {r["cer_generated"]:.2%} | {r["delta_cer"]:.2%} | {r["wer_generated"]:.2%} | {r["eog_rate"]:.0%} | '+ ' | '.join(vv)+' |')
    lines += ['',f'Real-image CER on these fixed lines: **{rows[0]["cer_real"]:.2%}**. '
        'Step 4900 minimizes logged validation loss (0.270020), compared with 0.270393 at step 5000. '
        'This loss improvement does not identify the best autonomous generator.','',
        f'Paired line-bootstrap 95% interval for CER(4550) − CER(5000): '
        f'**[{low*100:.2f}, {high*100:.2f}] percentage points** (10,000 resamples, seed 24). '
        'The interval is descriptive and does not account for selection across 100 checkpoints.','',
        'All checkpoint comparisons above use the same 10 lines / 275 characters. '
        'FID/BFID/KID are very unstable at this sample size, and numerical covariance warnings occurred. '
        'HWD uses page-based groups rather than verified writers. Confirm on a larger fixed validation set.','',
        '![Checkpoint curves](checkpoint-curves.png)','',
        'The requested step-5000 checkpoint also has a separate 20-preview analysis in `../analysis-final/`. '
        'Those scores use a different sample set and should not be mixed with this table.','',
        'Metric definitions, sources and runnable commands are in the project-root `EVALUATION.md`.']
    final_path=run/'analysis-final/summary.json'
    if final_path.exists():
        lines += ['', '## Requested checkpoint: all 20 previews', '', '```json', final_path.read_text(), '```']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(out/'report.md')


if __name__=='__main__':
    main()
