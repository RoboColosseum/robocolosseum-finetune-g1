"""Group single-episode review exports into homogeneous LeRobot datasets, without re-encoding."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise ValueError('Output already exists; do not overwrite a prior export')
    rows = json.loads(args.selection.read_text())['episodes_list']
    args.output.mkdir(parents=True, exist_ok=args.resume)
    all_rows, group_info = [], []
    for family in sorted({r['family'] for r in rows}):
        selected = [r for r in rows if r['family'] == family]
        roots = [args.source / r['dataset_path'] for r in selected]
        # Never silently pad/resize or merge incompatible feature shapes.
        signatures = []
        for root in roots:
            info = json.loads((root/'meta/info.json').read_text())
            signatures.append({k:{n:v[n] for n in ('dtype','shape','names') if n in v}
                               for k,v in info['features'].items()})
        assert all(s == signatures[0] for s in signatures), family
        target = args.output / family
        if not target.exists():
            aggregate_datasets(repo_ids=['local/'+p.name for p in roots],
                roots=roots, aggr_repo_id='local/g01-preview-'+family, aggr_root=target,
                concatenate_videos=False, concatenate_data=False)
        merged = pa.concat_tables([pq.read_table(p) for p in sorted((target/'data').rglob('*.parquet'))])
        info = json.loads((target/'meta/info.json').read_text())
        assert info['total_episodes'] == len(selected)
        assert info['total_frames'] == sum(r['frames'] for r in selected)
        tasks = pq.read_table(target/'meta/tasks.parquet').to_pandas()
        task_map = {int(v['task_index']):str(k) for k,v in tasks.iterrows()}
        reader = LeRobotDataset(repo_id='local/g01-preview-'+family, root=target,
                               video_backend='pyav', download_videos=False)
        offset = 0
        group_rows = []
        for episode_index, (row, source) in enumerate(zip(selected, roots)):
            original = pa.concat_tables([pq.read_table(p) for p in sorted((source/'data').rglob('*.parquet'))])
            actual = merged.filter(pa.compute.equal(merged['episode_index'], episode_index))
            assert actual.num_rows == original.num_rows == row['frames']
            for column in original.column_names:
                if column not in ('index','episode_index','task_index'):
                    assert actual[column].to_pylist() == original[column].to_pylist(), (family, episode_index, column)
            assert actual['index'].to_pylist() == list(range(offset, offset+row['frames']))
            task_indices = set(actual['task_index'].to_pylist())
            assert len(task_indices) == 1
            task_index = int(next(iter(task_indices)))
            manifest = json.loads((source/'source/manifest.json').read_text())
            task_text = task_map[task_index]
            assert task_text == manifest['config']['task']
            for i in sorted({offset, offset+row['frames']-1}):
                frame = reader[i]
                assert int(frame['episode_index']) == episode_index
                assert int(frame['task_index']) == task_index
                assert tuple(frame['action'].shape) == (23,)
                for key,feature in info['features'].items():
                    if feature['dtype']=='video':
                        assert list(frame[key].shape) == [3, feature['shape'][0], feature['shape'][1]]
            provenance = target/'source'/f'episode_{episode_index:06d}'
            provenance.mkdir(parents=True, exist_ok=True)
            for name in ('manifest.json','events.jsonl','summary.json','export.json'):
                if (source/'source'/name).exists():
                    shutil.copyfile(source/'source'/name, provenance/name)
            schema = manifest.get('policy_schema',{})
            entry = dict(model_family=family, model=row['model'],
                checkpoint=schema.get('model_path',manifest['config'].get('model_path','')),
                dataset_path=family, episode_index=episode_index,
                task=row['task'],task_index=task_index,task_instruction=task_text,
                frames=row['frames'],missing_responses=row['missing_responses'],
                source_episode=source.name,
                action_chunk_length=schema['action_shape'][0],
                head_width=info['features']['observation.images.head']['shape'][1])
            group_rows.append(entry)
            all_rows.append(entry)
            offset += row['frames']
        for path, entries in [(target/'episodes.csv', group_rows)]:
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
        original_videos=sorted(sha(p) for root in roots for p in (root/'videos').rglob('*.mp4'))
        merged_videos=sorted(sha(p) for p in (target/'videos').rglob('*.mp4'))
        assert original_videos == merged_videos, family
        group_info.append(dict(model_family=family,dataset_path=family,episodes=len(selected),frames=offset,
            tasks=task_map,action_chunk_length=group_rows[0]['action_chunk_length'],head_width=group_rows[0]['head_width'],
            all_source_columns_equal=True,video_bytes_unchanged=True,all_episode_first_last_read=True))
    with (args.output/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    summary=dict(format='LeRobot v3',datasets=group_info,episodes=len(all_rows),frames=sum(x['frames'] for x in all_rows),
                 note='episode_index and task_index are local to each model-family dataset; video bytes are unchanged')
    (args.output/'index.json').write_text(json.dumps(summary,indent=2)+'\n')
    files={str(p.relative_to(args.output)):dict(size=p.stat().st_size,sha256=sha(p))
           for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'file-manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
