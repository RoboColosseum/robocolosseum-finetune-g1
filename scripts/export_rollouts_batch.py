"""Resumable, create-only LeRobot review export for a pinned rollout selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def validate(source, output):
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from export_rollout_lerobot import read_trace
    manifest, requests, responses = read_trace(source)
    table = pq.read_table(sorted((output / 'data').rglob('*.parquet')))
    info = json.loads((output / 'meta/info.json').read_text())
    assert info['total_episodes'] == 1
    assert info['total_frames'] == table.num_rows == len(requests)
    assert table.column('episode_index').to_pylist() == [0] * len(requests)
    shape = tuple(manifest['policy_schema']['action_shape'])
    cols = table.to_pydict()
    missing = 0
    for i, request in enumerate(requests):
        for key in ('request_id', 'wall_unix_ns', 'monotonic_s', 'camera_time', 'state_time'):
            assert np.asarray(cols['source.' + key][i]).reshape(-1)[0] == request[key], (key, i)
        response = responses.get(request['request_id'])
        assert bool(np.asarray(cols['policy.response_valid'][i]).reshape(-1)[0]) == (response is not None)
        expected = (np.asarray(response['result']['actions'], dtype=np.float32)
                    if response is not None else np.zeros(shape, dtype=np.float32))
        assert np.array_equal(np.asarray(cols['policy.action_chunk'][i], dtype=np.float32), expected.ravel())
        assert np.array_equal(np.asarray(cols['action'][i], dtype=np.float32), expected[0])
        missing += response is None
    videos = []
    for key, feature in info['features'].items():
        if feature['dtype'] != 'video':
            continue
        paths = sorted((output / 'videos' / key).rglob('*.mp4'))
        assert len(paths) == 1, (key, paths)
        with av.open(str(paths[0])) as container:
            stream = container.streams.video[0]
            assert stream.frames == len(requests), (key, stream.frames, len(requests))
            frame = next(container.decode(video=0))
            assert [frame.height, frame.width, 3] == feature['shape']
        videos.append(str(paths[0].relative_to(output)))
    assert len(videos) == 3, videos
    files = {str(p.relative_to(output)): dict(size=p.stat().st_size,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'validation.json'}
    return dict(frames=len(requests), missing_responses=missing, videos=videos,
                files=files,
                source_events_sha256=hashlib.sha256((source / 'events.jsonl').read_bytes()).hexdigest(),
                source_manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
                validation='Parquet rows, all action chunks and source clocks, masks, video frame counts and decoded first frames')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    from export_rollout_lerobot import export
    selection = json.loads(args.selection.read_text())
    assert selection['episodes'] == 653 and not selection['errors']
    args.output.mkdir(parents=True, exist_ok=True)
    selection_digest = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    selection_copy = args.output / 'selection.json'
    if selection_copy.exists():
        assert hashlib.sha256(selection_copy.read_bytes()).hexdigest() == selection_digest
    else:
        shutil.copyfile(args.selection, selection_copy)
    rows = []
    suffix = '' if args.shards == 1 else '-' + str(args.shard)
    index_path = args.output / ('index' + suffix + '.json')
    status_path = args.output / ('status' + suffix + '.json')
    for item in selection['rows'][args.shard::args.shards]:
        if args.limit is not None and len(rows) >= args.limit:
            break
        source = args.source_root / item['path']
        assert source.resolve().is_relative_to(args.source_root.resolve())
        assert json.loads((source / 'manifest.json').read_text())['mode'] == 'execute'
        episode_id = source.name + '-' + hashlib.sha256(item['path'].encode()).hexdigest()[:8]
        target = args.output / 'episodes' / episode_id
        result = dict(source_path=item['path'], dataset_path='episodes/' + episode_id)
        verified = target / 'source/validation.json'
        if verified.exists():
            result.update(json.loads(verified.read_text()), status='complete')
        else:
            while shutil.disk_usage(args.output).free < 4 * 1024**3:
                write_json(status_path, dict(state='waiting_for_disk', complete=sum(r['status']=='complete' for r in rows), current=item['path']))
                time.sleep(20)
            staging = args.output / '.staging' / episode_id
            staging.parent.mkdir(exist_ok=True)
            start = time.monotonic()
            try:
                if target.exists():
                    raise ValueError('Unvalidated output exists; inspect before resuming')
                if not staging.exists():
                    export(source, staging, 'local/' + episode_id, 3, streaming_encoding=True)
                validation = validate(source, staging)
                validation['elapsed_seconds'] = round(time.monotonic() - start, 3)
                write_json(staging / 'source/validation.json', validation)
                target.parent.mkdir(exist_ok=True)
                staging.rename(target)
                result.update(validation, status='complete')
            except Exception as exc:
                result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        rows.append(result)
        write_json(index_path, dict(selection_sha256=selection_digest, episodes_requested=653, completed=sum(r['status']=='complete' for r in rows), failed=sum(r['status']=='failed' for r in rows), frames=sum(r.get('frames',0) for r in rows), episodes=rows))
        write_json(status_path, dict(state='running', processed=len(rows), completed=sum(r['status']=='complete' for r in rows), failed=sum(r['status']=='failed' for r in rows), current=item['path']))
        print(json.dumps(dict(processed=len(rows), **result)), flush=True)
    write_json(status_path, dict(state='finished', processed=len(rows), completed=sum(r['status']=='complete' for r in rows), failed=sum(r['status']=='failed' for r in rows)))


if __name__ == '__main__':
    main()
