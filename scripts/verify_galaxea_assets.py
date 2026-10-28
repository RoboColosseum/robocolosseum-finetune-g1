"""Verify every pinned Galaxea asset against its upstream Git or LFS digest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    inventory = json.loads((ROOT / 'dependencies/galaxea-source-inventory.json').read_text())
    checked = []
    for item in inventory['files']:
        path = ROOT / 'models/jasonren-g1' / item['rfilename']
        result = dict(path=item['rfilename'], ok=False)
        if path.is_file():
            size = path.stat().st_size
            sha256 = hashlib.sha256()
            git_hash = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                    sha256.update(block)
                    if 'lfs' not in item:
                        git_hash.update(block)
            actual = sha256.hexdigest() if 'lfs' in item else git_hash.hexdigest()
            expected = item['lfs']['sha256'] if 'lfs' in item else item['blobId']
            result.update(size=size, sha256=sha256.hexdigest(),
                          ok=size == item['size'] and actual == expected)
        checked.append(result)
        print(result['path'], 'VERIFIED' if result['ok'] else 'MISSING / MISMATCH', flush=True)
    report = dict(repo=inventory['repo'], revision=inventory['revision'], files=checked,
                  ok=all(f['ok'] for f in checked))
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/galaxea-all-files-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['ok']:
        raise SystemExit('Some Galaxea assets are missing or fail checksum validation')


if __name__ == '__main__':
    main()
