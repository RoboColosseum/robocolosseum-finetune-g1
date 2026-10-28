"""Create nonduplicating model and historical-rollout directory views."""
import json
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_deploy.catalog import catalog


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f'Existing link points elsewhere: {destination}')
        return
    if destination.exists():
        raise RuntimeError(f'Will not overwrite existing directory: {destination}')
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)


def main():
    c = catalog()
    for key, spec in c['models'].items():
        link(ROOT / spec['source_path'], ROOT / spec['model_path'])
    legacy = ROOT / 'models/groot-n17-g1-carry-box-40k'
    if legacy.exists():
        link(legacy, ROOT / 'models/by-family/groot/carry-box-40k')
    rollouts = ROOT / 'outputs/rollouts'
    if rollouts.exists():
        for run in rollouts.iterdir():
            manifest = run / 'manifest.json'
            if not manifest.is_file():
                continue
            data = json.loads(manifest.read_text())
            cfg = data.get('config', {})
            schema = data.get('policy_schema') or {}
            key = cfg.get('model_key', schema.get('model_key'))
            if key in c['models']:
                spec = c['models'][key]
                family, task = spec['family'], spec['task_slug']
            elif cfg.get('task') in {'plate into rack', 'load clothes washing machine', 'carrying box'}:
                family = 'groot'
                task = {'plate into rack':'plate-into-rack', 'load clothes washing machine':'load-clothes-washing-machine',
                        'carrying box':'carry-box'}[cfg['task']]
            else:
                family, task = 'unclassified', 'unknown'
            link(run, rollouts / 'by-model' / family / task / run.name)
    print('Classification links ready; original downloads and recordings are unchanged.')


if __name__ == '__main__':
    main()
