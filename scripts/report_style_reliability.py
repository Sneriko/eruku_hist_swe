"""Summarize fixed-style transfer trials and build paired visual review sheets."""
import collections
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    out = Path('output/testsuite-linear4250-style-reliability').resolve()
    pairs = json.loads((out/'pairs.json').read_text())
    results = [json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    selection = json.loads((out/'selection.json').read_text())
    lookup = {(r['group'], r['example']): r for r in results}
    ratings_path = out/'visual-ratings.json'
    ratings = json.loads(ratings_path.read_text()) if ratings_path.exists() else {}
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    def passed(r):
        return bool(r.get('file')) and r['eog_reached'] and not r.get('near_blank', False)
    def img(path, width=1320, height=90):
        with Image.open(path) as source:
            image = source.convert('RGB')
        image.thumbnail((width, height))
        return image
    summaries, baselines = [], []
    review = out/'review'
    review.mkdir(exist_ok=True)
    parts = ['<!doctype html><meta charset="utf-8"><title>Fixed style reliability</title>',
             '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid>div{min-width:0}article{border-top:1px solid #aaa;margin-top:20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
             '<h1>Does a successful style reference generalize to other texts?</h1>',
             '<p>Ten visually selected references, each tested on ten other texts from the same page. An alternate, unselected reference from that page generates the identical texts. Original successful texts are repeated separately. Same checkpoint, normalization, CFG 1.375, float32 and seed 24. Ten new texts per style were chosen before inference, spanning short, medium and long lines where available. Originals are excluded from held-out totals.</p>',
             '<p>Automatic pass means EOG reached and nonblank, not correct transcription. Visual grades, where provided, are subjective: usable = readable and substantially follows text; partial = recognizable but clear omissions/distortions; failed = blank, degenerate, runaway or largely unreadable.</p>']
    for s in selection:
        group = s['group']
        records = [r for r in pairs if r['group'] == group]
        if any((group, r['example']) not in lookup for r in records):
            continue
        held = [r for r in records if r['trial'] == 'held-out']
        summary = dict(group=group, original_example=s['original_example'])
        for role in ['selected', 'alternate']:
            subset = [r for r in held if r['reference_role'] == role]
            rr = [lookup[group, r['example']] for r in subset]
            summary[role+'_automatic_pass'] = sum(passed(r) for r in rr)
            summary[role+'_blank'] = sum(r.get('near_blank', False) for r in rr)
            summary[role+'_width_limit'] = sum(r.get('hit_width_limit', False) for r in rr)
            summary[role+'_visual'] = collections.Counter(ratings.get(group, {}).get(role, []))
        summaries.append(summary)
        baseline = lookup[group, 0]
        old = out/group/'references/original-generated.png'
        new = out/group/'00/generated.png'
        with Image.open(old) as image:
            a = np.array(image.convert('L'))
        if new.exists():
            with Image.open(new) as image:
                b = np.array(image.convert('L'))
            equal = bool(a.shape == b.shape and np.array_equal(a,b))
            mae = float(np.abs(a.astype(float)-b).mean()) if a.shape == b.shape else None
        else:
            equal, mae = False, None
        baselines.append(dict(group=group, automatic_pass=passed(baseline), original_width=a.shape[1],
                              repeat_width=baseline.get('width'), pixel_identical=equal, pixel_mae=mae))
        sheet = Image.new('RGB', (1500, 450 + 10*300), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((5,5),f"{group} / original example {s['original_example']:02}",font=font,fill='black')
        for j,(name,path) in enumerate([('selected style',out/group/'references/original-style.png'),
                                       ('original successful output',old),('original text repeated',new),
                                       ('alternate style',out/group/'references/alternate-style.png')]):
            draw.text((5,35+j*100),name,font=small,fill='black')
            if path.exists(): sheet.paste(img(path),(190,30+j*100))
        parts.append(f'<article id="{html.escape(group)}"><h2>{html.escape(group)}</h2><p>Held-out automatic pass: selected {summary["selected_automatic_pass"]}/10; alternate {summary["alternate_automatic_pass"]}/10.</p>')
        parts.append(f'<div class="grid"><div><h3>Selected reference</h3><p>{html.escape(s["selected_style"]["text"])}</p><img loading="lazy" src="{html.escape(group)}/references/original-style.png"><h4>Original successful output</h4><img loading="lazy" src="{html.escape(group)}/references/original-generated.png"><h4>Original text repeated</h4><img loading="lazy" src="{html.escape(group)}/00/generated.png"></div><div><h3>Alternate reference (not selected by output quality)</h3><p>{html.escape(s["alternate_style"]["text"])}</p><img loading="lazy" src="{html.escape(group)}/references/alternate-style.png"></div></div>')
        for trial in range(1,11):
            chosen = next(r for r in held if r['target_index']==trial and r['reference_role']=='selected')
            other = next(r for r in held if r['target_index']==trial and r['reference_role']=='alternate')
            top = 450+(trial-1)*300
            draw.text((5,top),f"{trial:02}: {chosen['text']}",font=font,fill='black')
            parts.append(f'<h3>New target {trial}: {html.escape(chosen["text"])}</h3><details><summary>Real target crop</summary><img loading="lazy" src="{html.escape(group)}/{chosen["example"]:02}/target.png"></details><div class="grid">')
            for j,r in enumerate([chosen,other]):
                result = lookup[group,r['example']]
                role = r['reference_role']
                label = f'{role}: EOG={result["eog_reached"]}, blank={result.get("near_blank")}, limit={result["hit_width_limit"]}'
                draw.text((5,top+30+j*125),label,font=small,fill='black')
                parts.append(f'<div><b>{html.escape(label)}</b>')
                if result.get('file'):
                    sheet.paste(img(result['file'],height=90),(170,top+55+j*125))
                    parts.append(f'<img loading="lazy" src="{html.escape(group)}/{r["example"]:02}/generated.png">')
                else: parts.append(f'<p>{html.escape(result.get("error","No image"))}</p>')
                parts.append('</div>')
            parts.append('</div>')
        sheet.save(review/f'{group}.jpg',quality=92)
        parts.append('</article>')
    summary = dict(groups=summaries, original_repeats=baselines)
    for role in ['selected', 'alternate']:
        summary[role+'_automatic_pass'] = sum(s[role+'_automatic_pass'] for s in summaries)
        summary[role+'_visual'] = sum((s[role+'_visual'] for s in summaries), collections.Counter())
    summary['total_new_texts_per_role'] = len(summaries)*10
    summary['baseline_repeats_pass'] = sum(r['automatic_pass'] for r in baselines)
    summary['baseline_repeats_pixel_identical'] = sum(r['pixel_identical'] for r in baselines)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (out/'results.csv').open('w',newline='') as f:
        fields = sorted(set().union(*(r.keys() for r in results)))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(results)
    (out/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
