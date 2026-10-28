"""Download the pinned model repository, verify weights, and cache base assets."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_deploy.catalog import catalog


def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--base-assets', action='store_true')
    p.add_argument('--family', help='Download and verify only this model family')
    a = p.parse_args()
    c = catalog()
    selected = {key: spec for key, spec in c['models'].items()
                if not a.family or spec['family'] == a.family}
    if not selected:
        p.error(f'Unknown or empty model family: {a.family}')
    if not a.verify_only:
        revisions = {}
        for spec in selected.values():
            revisions.setdefault(spec.get('revision', c['revision']), []).append(spec['folder'] + '/*')
        for revision, patterns in revisions.items():
            snapshot_download(c['repo'], revision=revision, allow_patterns=patterns,
                              local_dir=ROOT / 'models/jasonren-g1', max_workers=3)
    from scripts.organize_models import main as organize
    organize()
    report = {'repo': c['repo'], 'revision': c['revision'], 'models': {}}
    for key, spec in selected.items():
        files = spec.get('weight_files') or [dict(path='model.safetensors', size=spec['weight_bytes'], sha256=spec['weight_sha256'])]
        checked = []
        for entry in files:
            path = ROOT / spec['model_path'] / entry['path']
            item = {'path': str(path), 'ok': False}
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
                        digest.update(block)
                item.update(bytes=path.stat().st_size, sha256=digest.hexdigest())
                item['ok'] = item['bytes'] == entry['size'] and item['sha256'] == entry['sha256']
            checked.append(item)
        from g1_deploy.catalog import model_downloaded
        result = {'revision': spec.get('revision', c['revision']), 'files': checked,
                  'ok': all(f['ok'] for f in checked) and model_downloaded(spec)}
        report['models'][key] = result
        print(key, 'VERIFIED' if result['ok'] else 'MISSING / MISMATCH', flush=True)
    out = ROOT / 'outputs' / (f'{a.family}-models-verification.json' if a.family else 'all-models-verification.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    if not all(r['ok'] for r in report['models'].values()):
        raise SystemExit('Some checkpoints are missing or fail SHA256 verification')
    if a.base_assets:
        for repo in ('allenai/MolmoAct2', 'nvidia/GR00T-N1.7-3B', 'nvidia/Cosmos-Reason2-2B'):
            snapshot_download(repo, max_workers=2)


if __name__ == '__main__':
    main()
