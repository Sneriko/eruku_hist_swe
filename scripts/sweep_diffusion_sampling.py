"""Matched step-count/CFG sweep over short, medium and long handwriting lines."""
import argparse
import collections
import csv
import html
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont


def prepare(checkpoint, out):
    saved = json.loads((checkpoint/'samples/samples.json').read_text())
    suite = json.loads(Path('output/testsuite-page-diffusion-step4250/pairs.json').read_text())
    chosen = []
    for bucket, ids in [('short', [3,15,18,19,20,29]), ('medium',[0,4,7,9,13,14,24,25]),
                        ('long',[12,21,38,45])]:
        for i in ids:
            chosen.append((bucket, saved[i], checkpoint/f'samples/{i:02}-style.png',
                           checkpoint/f'samples/{i:02}-target.png', f'validation preview {i:02}'))
    for bucket, group, i in [('short','1654_(R0001308)',0),('short','1716_(A0017151)',0),
                              ('long','1654_(R0001308)',1),('long','1654_(R0001308)',2),
                              ('long','1871_(K0017448)',1),('long','1871_(K0017448)',4)]:
        r = next(r for r in suite if (r['group'],r['example']) == (group,i))
        chosen.append((bucket,r,Path(r['style_path']),Path(r['target_path']),f'test suite {group}/{i:02}'))
    chosen.sort(key=lambda item: ['short','medium','long'].index(item[0]))
    cases = []
    for i,(bucket,r,style,target,source) in enumerate(chosen):
        directory=out/'references'/f'{i:02}'
        directory.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(style,directory/'style.png')
        shutil.copyfile(target,directory/'target.png')
        (directory/'text.txt').write_text(f"Target: {r['text']}\nStyle: {r['style_text']}\n")
        cases.append(dict(case=i,bucket=bucket,source=source,text=r['text'],style_text=r['style_text'],
                          word_count=len(r['text'].split()),style_path=str(directory/'style.png'),
                          target_path=str(directory/'target.png')))
    assert collections.Counter(r['bucket'] for r in cases)=={'short':8,'medium':8,'long':8}
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    return cases


def gallery(out,cases,results,settings):
    configs=[(steps,cfg) for steps in settings['steps'] for cfg in settings['cfg']]
    lookup={(r['case'],r['seed'],r['steps'],r['cfg']):r for r in results}
    parts=['<!doctype html><meta charset="utf-8"><title>Diffusion steps and CFG sweep</title>',
           '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%;height:auto}article{border-top:1px solid #aaa;padding:18px 0}.scroll{overflow-x:auto}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:8px;vertical-align:top}td{min-width:500px}small{display:block}.baseline{background:#edf6ff}</style>',
           '<h1>Diffusion sampling steps × CFG</h1>',
           f'<p>Checkpoint: {html.escape(settings["checkpoint"])}. Selected by lowest logged validation loss (0.453996 at step 4000). Float32, normalized 128px lines, width cap 4096. Same pair and seed in every column; initial diffusion noise and VAE style draw are matched. Click a generated image for full resolution.</p>',
           f'<p>Eight short (1–3 words), eight medium (4–8), eight long (9–18). CFG 1.375 / 20 steps is the training-preview baseline. Seeds: {settings["seeds"]}. EOG/blankness are diagnostics, not text accuracy.</p>',
           f'<p>Saved outputs: {len(results)}/{len(cases)*len(configs)*len(settings["seeds"])}</p>']
    for case in cases:
        i=case['case']
        parts.extend([f'<article id="case-{i:02}"><h2>{i:02} · {case["bucket"]} · {case["word_count"]} words</h2>',
                      f'<p>{html.escape(case["source"])}</p><p>Style: {html.escape(case["style_text"])}</p><img loading="lazy" src="references/{i:02}/style.png">',
                      f'<p>Target: {html.escape(case["text"])}</p><img loading="lazy" src="references/{i:02}/target.png"><div class="scroll"><table><tr><th>Seed</th>'])
        for steps,cfg in configs:
            parts.append(f'<th>{steps} steps / CFG {cfg:g}</th>')
        parts.append('</tr>')
        for seed in settings['seeds']:
            parts.append(f'<tr><th>{seed}</th>')
            for steps,cfg in configs:
                result=lookup.get((i,seed,steps,cfg))
                parts.append('<td class="baseline">' if (steps,cfg)==(20,1.375) else '<td>')
                if result:
                    if result['file']:
                        path=html.escape(result['file'])
                        parts.append(f'<a href="{path}"><img loading="lazy" src="{path}"></a>')
                    else: parts.append(html.escape(result.get('error','No image')))
                    parts.append(f'<small>EOG {result["eog_reached"]} · blank {result.get("near_blank")} · width {result.get("width")}</small>')
                else:parts.append('Pending')
                parts.append('</td>')
            parts.append('</tr>')
        parts.append('</table></div></article>')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf-8')


def summarize(out,cases,results,settings):
    rows=[]
    for bucket in ['short','medium','long','all']:
        ids={r['case'] for r in cases if bucket=='all' or r['bucket']==bucket}
        for steps in settings['steps']:
            for cfg in settings['cfg']:
                subset=[r for r in results if r['case'] in ids and r['steps']==steps and r['cfg']==cfg]
                rows.append(dict(bucket=bucket,steps=steps,cfg=cfg,n=len(subset),
                                 eog=sum(r['eog_reached'] for r in subset),blank=sum(r.get('near_blank',False) for r in subset),
                                 width_limit=sum(r['hit_width_limit'] for r in subset),
                                 errors=sum('error' in r for r in subset)))
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in results))));w.writeheader();w.writerows(results)
    # One full-width sheet per prompt/seed makes lettering easier to inspect than thumbnails.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    review=out/'review';review.mkdir(exist_ok=True)
    lookup={(r['case'],r['seed'],r['steps'],r['cfg']):r for r in results}
    configs=[(steps,cfg) for steps in settings['steps'] for cfg in settings['cfg']]
    for case in cases:
        for seed in settings['seeds']:
            sheet=Image.new('RGB',(1800,1000),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,5),f"{case['case']:02} {case['bucket']} seed {seed}: {case['text']}",font=font,fill='black')
            for row,(steps,cfg) in enumerate(configs):
                result=lookup[case['case'],seed,steps,cfg]
                draw.text((8,45+row*155),f'{steps} steps / CFG {cfg:g}',font=font,fill='black')
                if result['file']:
                    with Image.open(out/result['file']) as source:image=source.convert('RGB')
                    image.thumbnail((1570,128));sheet.paste(image,(225,40+row*155))
            sheet.save(review/f"{case['case']:02}-seed-{seed}.jpg",quality=94)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[24])
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    settings=dict(checkpoint=str(args.checkpoint.resolve()),steps=[20,40],cfg=[1.0,1.25,1.375],seeds=args.seeds,
                  precision='float32',batch_size=12,checkpoint_validation_loss=0.453996)
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Existing sweep settings differ')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    cases=json.loads((out/'cases.json').read_text()) if (out/'cases.json').exists() else prepare(args.checkpoint.resolve(),out)
    results_path=out/'results.jsonl'
    results=[json.loads(s) for s in results_path.read_text().splitlines()] if results_path.exists() else []
    gallery(out,cases,results,settings)
    import torch
    from historical_swedish.checkpoints import load_checkpoint,head_options
    from eruku_continuous_inf import DDPCompatibleEmuru
    from scripts.generate_testsuite_page_examples import render_batch,accelerate_diffusion_sampling
    metadata,state=load_checkpoint(args.checkpoint)
    model=DDPCompatibleEmuru(metadata['t5_config'],metadata['vae'],ocr_checkpoint=None,
                            image_height=metadata['height'],**head_options(metadata))
    model.load_state_dict(state);del state
    model.to('cuda').eval();accelerate_diffusion_sampling(model)
    done={(r['case'],r['seed'],r['steps'],r['cfg']) for r in results}
    expected=len(cases)*len(settings['steps'])*len(settings['cfg'])*len(settings['seeds'])
    verified=False
    with results_path.open('a',buffering=1) as handle:
        for steps in settings['steps']:
            model.diffusion_steps=steps
            for cfg in settings['cfg']:
                for seed in settings['seeds']:
                    pending=[r for r in cases if (r['case'],seed,steps,cfg) not in done]
                    for start in range(0,len(pending),12):
                        batch=pending[start:start+12];began=time.monotonic()
                        outputs=render_batch(model,batch,seed,verify=not verified,cfg_scale=cfg)
                        verified=True
                        for case,(image,details) in zip(batch,outputs):
                            name=f"steps-{steps}/cfg-{cfg:g}/seed-{seed}/{case['case']:02}.png"
                            if image is not None:
                                destination=out/name;destination.parent.mkdir(parents=True,exist_ok=True);image.save(destination)
                            record=dict(case=case['case'],bucket=case['bucket'],seed=seed,steps=steps,cfg=cfg,
                                        file=name if image is not None else None,**details)
                            handle.write(json.dumps(record)+'\n');results.append(record)
                        gallery(out,cases,results,settings)
                        print(f'{len(results)}/{expected}: steps={steps}, CFG={cfg}, seed={seed}, batch={len(batch)}, seconds={time.monotonic()-began:.1f}',flush=True)
    summarize(out,cases,results,settings)
    assert len(results)==expected
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=len(results),expected=expected)))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
