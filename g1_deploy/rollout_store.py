"""Local rollout inventory and guarded removal; no robot dependencies."""
import json
import os
from pathlib import Path
import shutil

from .recording import rollout_identity


def recording_processes():
    """Also protect command-line clients not owned by the web console."""
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'g1_deploy.wbc_client' in args:
                found.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Other users' clients cannot safely be ruled out.
            raise ValueError('Cannot check recording processes; deletion is unavailable')
    return found


def _read(path):
    if path.is_symlink():
        raise ValueError('Recording metadata must not be a symlink')
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Invalid recording metadata')
    return data


def inventory(root):
    root = Path(root)
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        path = Path(directory)
        dirs[:] = [d for d in dirs if not (path / d).is_symlink() and d != 'by-name']
        if 'manifest.json' not in files:
            continue
        dirs[:] = []
        try:
            manifest = _read(path / 'manifest.json')
            if manifest.get('format') != 'g01-rollout-v1':
                continue
            task, model = rollout_identity(manifest.get('config', {}), manifest.get('policy_schema'))
            summary = _read(path / 'summary.json') if (path / 'summary.json').exists() else {}
            size = 0
            for folder, children, names in os.walk(path, followlinks=False):
                children[:] = [n for n in children if not (Path(folder) / n).is_symlink()]
                for name in names:
                    f = Path(folder) / name
                    if not f.is_symlink():
                        size += f.stat().st_size
            rows.append(dict(id=path.relative_to(root).as_posix(), name=path.name,
                             task=task, model=model, mode=manifest.get('mode'),
                             started_at=manifest.get('start_unix_s', path.stat().st_mtime),
                             size_bytes=size, flush_complete=summary.get('flush_complete', False),
                             summary_present=bool(summary)))
        except (OSError, ValueError, TypeError):
            continue  # A recording may be creating its manifest during this scan.
    return sorted(rows, key=lambda r: r['started_at'], reverse=True)


def delete_rollout(root, rollout_id, confirmation):
    root = Path(root)
    if not isinstance(rollout_id, str) or not rollout_id:
        raise ValueError('Select a recording')
    parts = rollout_id.split('/')
    if any(p in ('', '.', '..') for p in parts) or Path(rollout_id).is_absolute():
        raise ValueError('Invalid recording path')
    if confirmation != parts[-1]:
        raise ValueError('Confirm the recording name before deleting')
    target = root
    if root.is_symlink():
        raise ValueError('Recording root must not be a symlink')
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise ValueError('Cannot delete through a symlink')
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Recording is outside the rollout directory')
    if not target.is_dir():
        raise ValueError('Recording no longer exists; refresh the list')
    if _read(target / 'manifest.json').get('format') != 'g01-rollout-v1':
        raise ValueError('Not a rollout recording')
    # Never treat a grouping folder as an episode, even if it has a manifest.
    if any(p != target / 'manifest.json' for p in target.rglob('manifest.json')):
        raise ValueError('Cannot delete a folder containing other recordings')
    aliases = [p for p in (root / 'by-name').glob('*')
               if p.is_symlink() and p.resolve() == target.resolve()]
    shutil.rmtree(target)
    for alias in aliases:
        alias.unlink(missing_ok=True)
    # These are generated snapshots; the web list always scans actual recordings.
    for name in ('index.json', 'index.csv'):
        (root / name).unlink(missing_ok=True)
    return {'deleted': rollout_id}
