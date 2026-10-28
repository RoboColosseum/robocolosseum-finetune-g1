"""Download and verify the official shared G05 processor, without base weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from huggingface_hub import snapshot_download
    inventory = json.loads((ROOT / 'dependencies/galaxea-processor-inventory.json').read_text())
    destination = ROOT / 'models/base/G05'
    snapshot_download(inventory['repo'], revision=inventory['revision'],
                      allow_patterns=[f['rfilename'] for f in inventory['files']],
                      local_dir=destination, max_workers=2)
    checked = []
    for item in inventory['files']:
        data = (destination / item['rfilename']).read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        if 'lfs' in item:
            valid_hash = sha256 == item['lfs']['sha256']
        else:
            blob = b'blob ' + str(len(data)).encode() + b'\0' + data
            valid_hash = hashlib.sha1(blob).hexdigest() == item['blobId']
        checked.append(dict(path=item['rfilename'], size=len(data), sha256=sha256,
                            ok=len(data) == item['size'] and valid_hash))
    report = dict(repo=inventory['repo'], revision=inventory['revision'], files=checked,
                  ok=all(f['ok'] for f in checked))
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/galaxea-processor-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['ok']:
        raise SystemExit('Processor checksum validation failed')
    print(f'Official G05 processor verified: {len(checked)} files. Training-host equivalence still requires validation.')


if __name__ == '__main__':
    main()
