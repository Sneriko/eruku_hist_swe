#!/usr/bin/env python3
"""Download paired FontSquare fine-tuning shards for Eruku replay.

Requires huggingface_hub (installed with requirements-historical.txt).
Downloads are resumable through the Hugging Face local download cache.
"""
import argparse
from pathlib import Path
import shlex

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = 'blowing-up-groundhogs/font-square-pairs-ft'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('/mnt/scratch/emuru_eruku_data/data/font-square-pairs-ft'))
    parser.add_argument('--num-shards', type=int, default=16,
                        help='Number of shards to download (default: 16)')
    parser.add_argument('--start-shard', type=int, default=0,
                        help='Zero-based position in the sorted shard list (default: 0)')
    parser.add_argument('--all', action='store_true', help='Download all shards after --start-shard')
    parser.add_argument('--revision', default='main', help='Dataset branch, tag or commit')
    parser.add_argument('--dry-run', action='store_true', help='Show files and size without downloading')
    args = parser.parse_args()
    if args.num_shards <= 0 or args.start_shard < 0:
        parser.error('--num-shards must be positive and --start-shard nonnegative')

    api = HfApi()
    # Pin one commit for discovery and download so files cannot change mid-run.
    revision = api.dataset_info(REPO_ID, revision=args.revision).sha
    files = sorted((entry for entry in api.list_repo_tree(
        REPO_ID, repo_type='dataset', revision=revision, recursive=True
    ) if getattr(entry, 'path', '').endswith('.tar')), key=lambda entry: entry.path)
    stop = None if args.all else args.start_shard + args.num_shards
    selected = files[args.start_shard:stop]
    if not selected:
        parser.error(f'No shards selected; dataset contains {len(files)} tar files')
    if not args.all and len(selected) < args.num_shards:
        parser.error(f'Only {len(selected)} shards remain; reduce --num-shards or use --all')

    destination = args.output_dir.expanduser().resolve()
    total = sum(entry.size for entry in selected)
    print(f'Dataset: {REPO_ID}\nCommit: {revision}')
    print(f'Selected {len(selected)}/{len(files)} shards ({total / 1024**3:.2f} GiB)')
    print(f'Destination: {destination}')
    for entry in selected:
        print(f'  {entry.path} ({entry.size / 1024**2:.1f} MiB)')
    if args.dry_run:
        return

    downloaded = []
    for index, entry in enumerate(selected, 1):
        print(f'Downloading {index}/{len(selected)}: {entry.path}', flush=True)
        path = hf_hub_download(repo_id=REPO_ID, repo_type='dataset',
                               filename=entry.path, revision=revision, local_dir=destination)
        downloaded.append(str(Path(path).resolve()))

    # Publish the list only once every requested shard is available.
    manifest = destination / 'shards.txt'
    temporary = destination / 'shards.txt.tmp'
    temporary.write_text('\n'.join(downloaded) + '\n', encoding='utf-8')
    temporary.replace(manifest)
    (destination / 'download-source.txt').write_text(
        f'repo_id={REPO_ID}\nrevision={revision}\n', encoding='utf-8')
    print('\nDownload complete. Add to your training command:')
    print(f'  --replay-train-pattern {shlex.quote(str(manifest))} --historical-probability 0.9')


if __name__ == '__main__':
    main()
