"""Prepare fixed-reference held-out text trials and same-page alternate controls."""
import json
from pathlib import Path
import random
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from historical_swedish.common import crop_polygon, normalize_background, normalize_text, parse_points, to_line_height
from historical_swedish.extract_page_lines import local_name
from scripts.generate_testsuite_page_examples import child


SELECTED = [('1670_jan-april', 1), ('1679_(R0002397)', 4), ('1690_(A0066756)', 2),
            ('1716_(A0017151)', 2), ('1755-1756_(C0112394)', 6), ('1775-1786_(A0068551)', 9),
            ('1792_(R0002231)', 5), ('1794-1795_(A0068546)', 3), ('1841_(Z0000017)', 6),
            ('1845-1857_(B0000011)', 4)]


def main():
    source = Path('output/testsuite-page-linear15k-step4250').resolve()
    out = Path('output/testsuite-linear4250-style-reliability').resolve()
    if (out / 'pairs.json').exists():
        raise FileExistsError('Experiment already prepared; reuse its fixed pairs')
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((source / 'pairs.json').read_text())
    pairs, selections = [], []
    for group, example in SELECTED:
        original = next(r for r in previous if (r['group'], r['example']) == (group, example))
        rng = random.Random(f'24:{group}:{example}')
        root = ET.parse(original['xml_path']).getroot()
        lines = []
        with Image.open(original['page_image']) as page:
            page = page.convert('RGB')
            for line in root.iter():
                if local_name(line.tag) != 'TextLine':
                    continue
                coords, equiv = child(line, 'Coords'), child(line, 'TextEquiv')
                unicode = child(equiv, 'Unicode') if equiv is not None else None
                if coords is None or unicode is None or not unicode.text:
                    continue
                text = normalize_text(unicode.text)
                if not 1 <= len(text.split()) <= 32:
                    continue
                points = parse_points(coords.get('points', ''))
                if len(points) < 3:
                    continue
                raw = to_line_height(crop_polygon(page, points, 8), 128, 4096)
                if raw is None or raw.width < 32:
                    continue
                lines.append((dict(line_id=line.get('id'), text=text, points=points), raw))
        exclude = {original['style_line']['line_id'], original['target_line']['line_id']}
        alternates = [(line, crop) for line, crop in lines
                      if line['line_id'] not in exclude and 3 <= len(line['text'].split()) <= 8]
        rng.shuffle(alternates)
        alternate, alternate_crop = min(alternates, key=lambda item: abs(len(item[0]['text'].split()) - len(original['style_text'].split())))
        exclude.add(alternate['line_id'])
        candidates = [(line, crop) for line, crop in lines if line['line_id'] not in exclude
                      and line['text'] not in (original['text'], original['style_text'], alternate['text'])]
        rng.shuffle(candidates)
        buckets = [[item for item in candidates if len(item[0]['text'].split()) <= 3],
                   [item for item in candidates if 4 <= len(item[0]['text'].split()) <= 8],
                   [item for item in candidates if len(item[0]['text'].split()) >= 9]]
        chosen, used_texts = [], set()
        for bucket, quota in zip(buckets, [3, 4, 3]):
            for item in bucket:
                if sum(1 for x in chosen if x in bucket) >= quota:
                    break
                if item[0]['text'] not in used_texts:
                    chosen.append(item)
                    used_texts.add(item[0]['text'])
        for item in candidates:
            if len(chosen) >= 10:
                break
            if item[0]['text'] not in used_texts:
                chosen.append(item)
                used_texts.add(item[0]['text'])
        assert len(chosen) == 10, (group, len(chosen))
        assets = out / group / 'references'
        assets.mkdir(parents=True, exist_ok=True)
        for name in ['style-raw.png', 'style.png', 'generated.png']:
            shutil.copyfile(source/group/f'{example:02}'/name, assets/f'original-{name}')
        alternate_crop.save(assets/'alternate-style-raw.png')
        normalize_background(alternate_crop).save(assets/'alternate-style.png')
        selections.append(dict(group=group, original_example=example, xml_path=original['xml_path'],
                               page_image=original['page_image'], selected_style=original['style_line'],
                               alternate_style=alternate, original_target=original['target_line'],
                               selection_reason='Visually readable and mostly faithful original output; no claim of character-perfect transcription',
                               alternate_selection='Unselected same-page 3–8-word reference, closest word count; no output-based filtering',
                               target_word_counts=[len(line['text'].split()) for line, _ in chosen]))
        targets = [(original['target_line'], None)] + chosen
        for trial, (target, raw) in enumerate(targets):
            for role_index, role in enumerate(['selected', 'alternate']):
                index = trial * 2 + role_index
                directory = out / group / f'{index:02}'
                directory.mkdir(parents=True, exist_ok=True)
                style = original['style_line'] if role == 'selected' else alternate
                for suffix in ['-raw.png', '.png']:
                    style_source = assets/('original-style' + suffix if role == 'selected' else 'alternate-style' + suffix)
                    shutil.copyfile(style_source, directory/('style' + suffix))
                if raw is None:
                    for name in ['target-raw.png', 'target.png']:
                        shutil.copyfile(source/group/f'{example:02}'/name, directory/name)
                else:
                    raw.save(directory/'target-raw.png')
                    normalize_background(raw).save(directory/'target.png')
                record = dict(group=group, example=index, original_example=example, reference_role=role,
                              trial='original-repeat' if trial == 0 else 'held-out', target_index=trial,
                              xml_path=original['xml_path'], page_image=original['page_image'],
                              style_line=style, target_line=target, style_text=style['text'], text=target['text'],
                              style_path=str(directory/'style.png'), target_path=str(directory/'target.png'),
                              generated_path=str(directory/'generated.png'), seed=24)
                assert style['line_id'] != target['line_id']
                (directory/'text.txt').write_text(f"Target: {target['text']}\nStyle: {style['text']}\nRole: {role}\nTrial: {record['trial']}\n")
                (directory/'source.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
                pairs.append(record)
        print(group, 'ten held-out word counts', selections[-1]['target_word_counts'], flush=True)
    (out/'pairs.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2))
    (out/'selection.json').write_text(json.dumps(selections, ensure_ascii=False, indent=2))
    assert len(pairs) == 220
    print('Prepared 100 selected-reference trials, 100 alternate-reference controls, and 20 original-text checks')


if __name__ == '__main__':
    main()
