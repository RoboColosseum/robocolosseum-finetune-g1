"""Finalize a pinned execute-rollout collection as minimal, schema-homogeneous LeRobot datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    import lerobot.datasets.aggregate as aggregate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from g1_deploy.recording import rollout_identity

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts',type=Path,required=True)
    parser.add_argument('--exports',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.audit.mkdir(parents=True,exist_ok=True)
    selection=json.loads(args.selection.read_text())
    assert selection['episodes']==653 and not selection['errors']
    groups={}
    for row in selection['rows']:
        source=args.rollouts/row['path']
        manifest=json.loads((source/'manifest.json').read_text())
        assert manifest['mode']=='execute'
        schema=manifest['policy_schema']
        task, model=rollout_identity(manifest['config'],schema)
        family=schema.get('family') or ('groot' if model.startswith(('groot','gr00t')) else model.split('-')[0])
        head_mode=schema.get('head_mode',manifest['config'].get('camera',{}).get('head_mode'))
        model_label=family
        if family=='groot' and task=='carry-box':
            model_label='groot-head1280' if head_mode=='stereo_full' else 'groot-head640'
        group=f'g1-{task}-{model_label}-evals'
        episode_id=source.name+'-'+hashlib.sha256(row['path'].encode()).hexdigest()[:8]
        groups.setdefault(group,[]).append(dict(source_path=row['path'],export_id=episode_id,
            model_family=family,model_name=model,task_name=task,
            checkpoint=schema.get('model_path',manifest['config'].get('model_path','')),
            action_chunk_length=schema['action_shape'][0],frames=row['requests'],
            task_instruction=manifest['config']['task']))
    write_json(args.audit/'selection.json',dict(episodes=653,groups=groups))
    # Videos are immutable derived exports on the same filesystem. Linking avoids
    # a second full copy on the deployment disk; original JSON rollouts are untouched.
    original_copy=aggregate.shutil.copy
    def copy_or_link(src,dst,*a,**kw):
        src,dst=Path(src),Path(dst)
        if src.suffix=='.mp4' and dst.resolve().is_relative_to(args.output.resolve()):
            assert src.resolve().is_relative_to(args.exports.resolve())
            os.link(src,dst)
            return str(dst)
        return original_copy(src,dst,*a,**kw)
    aggregate.shutil.copy=copy_or_link
    # LeRobot may emit bool or float arrays for boolean-feature statistics,
    # depending on the episode contents. Pandas/Arrow cannot append both in one
    # metadata column. Normalize only these statistics; data masks stay bool.
    original_append=aggregate.append_or_create_parquet_file
    active_roots=[]
    wall_bounds={}
    def append_with_boolean_stats(df,*a,**kw):
        columns=[k for k in df.columns if k.startswith('stats/policy.response_valid/')
                 and k.rsplit('/',1)[-1] in ('min','max','mean','std')]
        wall_columns=[k for k in ('stats/source.wall_unix_ns/min','stats/source.wall_unix_ns/max')
                      if k in df.columns]
        if columns or wall_columns:
            df=df.copy()
            for key in columns:
                df[key]=df[key].map(lambda value: np.asarray(value,dtype=np.float64))
            # Some one-frame exports describe integer timestamp statistics as
            # float arrays. Recompute exact int64 extrema from the stored data.
            for key in wall_columns:
                values=[]
                for episode_index in df['episode_index']:
                    episode_index=int(episode_index)
                    if episode_index not in wall_bounds:
                        paths=sorted((active_roots[episode_index]/'data').rglob('*.parquet'))
                        times=pq.read_table(paths,columns=['source.wall_unix_ns'])['source.wall_unix_ns'].to_pylist()
                        wall_bounds[episode_index]=(min(times),max(times))
                    value=wall_bounds[episode_index][0 if key.endswith('/min') else 1]
                    values.append(np.array([value],dtype=np.int64))
                df[key]=values
        return original_append(df,*a,**kw)
    aggregate.append_or_create_parquet_file=append_with_boolean_stats
    finished={}
    for name in groups:
        report=args.audit/(name+'-verification.json')
        if report.exists():finished[name]=json.loads(report.read_text())
    while len(finished)<len(groups):
        progressed=False
        for name,rows in groups.items():
            if name in finished:continue
            roots=[args.exports/'episodes'/r['export_id'] for r in rows]
            if not all((root/'source/validation.json').is_file() for root in roots):continue
            destination=args.output/name
            if destination.exists():raise ValueError(f'Unverified partial final dataset exists: {destination}')
            features=[json.loads((root/'meta/info.json').read_text())['features'] for root in roots]
            signatures=[{k:{a:v[a] for a in ('dtype','shape','names') if a in v} for k,v in f.items()} for f in features]
            assert all(s==signatures[0] for s in signatures),name
            active_roots[:]=roots
            wall_bounds.clear()
            aggregate.aggregate_datasets(repo_ids=['local/'+p.name for p in roots],roots=roots,
                aggr_repo_id='local/g01-'+name,aggr_root=destination,
                concatenate_videos=False,concatenate_data=False)
            stats_path=destination/'meta/stats.json'
            stats=json.loads(stats_path.read_text())
            stats['source.wall_unix_ns']['min']=[min(v[0] for v in wall_bounds.values())]
            stats['source.wall_unix_ns']['max']=[max(v[1] for v in wall_bounds.values())]
            write_json(stats_path,stats)
            info=json.loads((destination/'meta/info.json').read_text())
            assert info['total_episodes']==len(rows)
            assert info['total_frames']==sum(r['frames'] for r in rows)
            # Keep the requested model/task classification inside LeRobot's standard episode metadata files.
            for p in sorted((destination/'meta/episodes').rglob('*.parquet')):
                table=pq.read_table(p)
                indices=table['episode_index'].to_pylist()
                for key in ('model_family','model_name','checkpoint','task_name','action_chunk_length'):
                    values=pa.array([rows[i][key] for i in indices])
                    table=table.append_column(key,values)
                pq.write_table(table,p)
            tasks=pq.read_table(destination/'meta/tasks.parquet').to_pandas()
            task_map={int(v['task_index']):str(k) for k,v in tasks.iterrows()}
            data_paths=sorted((destination/'data').rglob('*.parquet'))
            assert len(data_paths)==len(rows)
            reader=LeRobotDataset(repo_id='local/g01-'+name,root=destination,video_backend='pyav',download_videos=False)
            offset=missing_responses=0
            for episode_index,(row,source,p) in enumerate(zip(rows,roots,data_paths)):
                original=pa.concat_tables([pq.read_table(x) for x in sorted((source/'data').rglob('*.parquet'))])
                actual=pq.read_table(p)
                assert actual.num_rows==original.num_rows==row['frames']
                assert actual['episode_index'].to_pylist()==[episode_index]*row['frames']
                assert actual['index'].to_pylist()==list(range(offset,offset+row['frames']))
                for column in original.column_names:
                    if column not in ('index','episode_index','task_index'):
                        assert actual[column].to_pylist()==original[column].to_pylist(),(name,episode_index,column)
                ti=set(actual['task_index'].to_pylist())
                assert len(ti)==1 and task_map[next(iter(ti))]==row['task_instruction']
                for index in sorted({offset,offset+row['frames']-1}):
                    frame=reader[index]
                    assert int(frame['episode_index'])==episode_index
                    assert tuple(frame['action'].shape)==(23,)
                    for key,feature in info['features'].items():
                        if feature['dtype']=='video':
                            assert list(frame[key].shape)==[3,feature['shape'][0],feature['shape'][1]]
                missing_responses+=json.loads((source/'source/validation.json').read_text())['missing_responses']
                offset+=row['frames']
            source_video_hashes=sorted(digest(p) for root in roots for p in (root/'videos').rglob('*.mp4'))
            target_video_hashes=sorted(digest(p) for p in (destination/'videos').rglob('*.mp4'))
            assert source_video_hashes==target_video_hashes
            assert {p.name for p in destination.iterdir()}=={'meta','data','videos'}
            file_manifest={str(p.relative_to(destination)):dict(size=p.stat().st_size,sha256=digest(p))
                           for p in sorted(destination.rglob('*')) if p.is_file()}
            write_json(args.audit/(name+'-files.json'),file_manifest)
            result=dict(dataset=name,episodes=len(rows),frames=offset,missing_responses=missing_responses,
                all_source_values_equal=True,all_episodes_first_last_read=True,video_bytes_unchanged=True,
                files=len(file_manifest),bytes=sum(x['size'] for x in file_manifest.values()),
                head_width=info['features']['observation.images.head']['shape'][1],action_chunk_length=rows[0]['action_chunk_length'])
            write_json(args.audit/(name+'-verification.json'),result)
            finished[name]=result
            progressed=True
            print(json.dumps(result),flush=True)
        write_json(args.audit/'status.json',dict(state='complete' if len(finished)==len(groups) else 'waiting_for_exports',
            groups_completed=len(finished),groups_total=len(groups),episodes=sum(r['episodes'] for r in finished.values()),
            frames=sum(r['frames'] for r in finished.values()),datasets=finished))
        if not progressed:time.sleep(15)
    assert sum(r['episodes'] for r in finished.values())==653
    assert sum(r['frames'] for r in finished.values())==selection['requests']


if __name__=='__main__':main()
