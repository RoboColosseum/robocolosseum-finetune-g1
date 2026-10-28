"""Export a recorded inference trace for review, without robot connections or Hub uploads.

One dataset row per policy request, with all raw action chunks and source clocks.
LeRobot's nominal FPS is a playback timeline, not a claim of camera capture rate.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import shutil


def read_trace(source):
    manifest = json.loads((source / 'manifest.json').read_text())
    requests, responses = [], {}
    for line in (source / 'events.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e['event'] == 'policy_request':
            requests.append(e)
        elif e['event'] == 'policy_response':
            if e['request_id'] in responses:
                raise ValueError('Duplicate response ID')
            responses[e['request_id']] = e
    if not requests or len({e['request_id'] for e in requests}) != len(requests):
        raise ValueError('Missing requests or duplicate request IDs')
    if any(b['monotonic_s'] <= a['monotonic_s'] for a, b in zip(requests, requests[1:])):
        raise ValueError('Request timestamps are not strictly increasing')
    for e in requests:
        p = (source / e['payload_file']).resolve()
        if not p.is_relative_to(source.resolve()) or not p.is_file():
            raise ValueError('Missing or unsafe request payload')
    return manifest, requests, responses


def export(source, output, repo_id, fps, *, streaming_encoding=False):
    import numpy as np
    from PIL import Image
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    manifest, requests, responses = read_trace(source)
    shape = tuple(manifest['policy_schema']['action_shape'])
    names = manifest['policy_schema']['action_names']
    first = json.loads((source / requests[0]['payload_file']).read_text())
    def vector_feature(size, dtype='float32', names=None):
        return dict(dtype=dtype, shape=(size,), names=names)
    features = {
        'observation.state': vector_feature(29),
        'observation.velocity': vector_feature(29),
        'observation.state.wbc': vector_feature(23),
        'action': vector_feature(shape[1], names=names),
        'policy.action_chunk': vector_feature(shape[0] * shape[1]),
        'policy.response_valid': vector_feature(1, 'bool'),
        'source.request_id': vector_feature(1, 'int64'),
        'source.wall_unix_ns': vector_feature(1, 'int64'),
    }
    for name in ('monotonic_s', 'camera_time', 'state_time', 'relative_time_s'):
        features['source.' + name] = vector_feature(1, 'float64')
    for name, encoded in first['images'].items():
        im = Image.open(io.BytesIO(base64.b64decode(encoded)))
        features['observation.images.' + name] = dict(
            dtype='video', shape=(im.height, im.width, 3), names=['height', 'width', 'channels'])
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=fps,
        robot_type='unitree_g1', features=features, image_writer_threads=2, video_backend='pyav',
        rgb_encoder=RGBEncoderConfig(vcodec='h264', crf=18, preset='fast'), encoder_threads=2,
        streaming_encoding=streaming_encoding,
        # This is an offline export: buffering must never drop a source frame
        # when CPU contention temporarily stalls the video encoder.
        encoder_queue_maxsize=max(30, len(requests)) if streaming_encoding else 30)
    missing = []
    for e in requests:
        payload = json.loads((source / e['payload_file']).read_text())
        response = responses.get(e['request_id'])
        valid = response is not None
        if valid:
            actions = np.asarray(response['result']['actions'], dtype=np.float32)
            if actions.shape != shape or not np.isfinite(actions).all():
                raise ValueError('Invalid recorded action chunk')
        else:
            missing.append(e['request_id'])
            # Explicitly masked storage placeholder; never a claimed model output.
            actions = np.zeros(shape, dtype=np.float32)
        frame = {
            'observation.state': np.asarray(payload['state'], dtype=np.float32),
            'observation.velocity': np.asarray(e['dq'], dtype=np.float32),
            'observation.state.wbc': np.asarray(payload['wbc_state'], dtype=np.float32),
            'action': actions[0].copy(), 'policy.action_chunk': actions.reshape(-1),
            'policy.response_valid': np.array([valid], dtype=bool),
            'task': payload['task'],
        }
        for name in ('request_id', 'wall_unix_ns'):
            frame['source.' + name] = np.array([e[name]], dtype=np.int64)
        for name in ('monotonic_s', 'camera_time', 'state_time'):
            frame['source.' + name] = np.array([e[name]], dtype=np.float64)
        frame['source.relative_time_s'] = np.array([e['monotonic_s'] - requests[0]['monotonic_s']], dtype=np.float64)
        for name, encoded in payload['images'].items():
            frame['observation.images.' + name] = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB'))
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()
    provenance = output / 'source'
    provenance.mkdir()
    for name in ('manifest.json', 'events.jsonl', 'summary.json'):
        if (source / name).exists():
            shutil.copyfile(source / name, provenance / name)
    notes = dict(purpose='review only; not aligned training demonstrations',
        source_rollout=source.name, rows=len(requests), raw_action_chunk_shape=shape,
        action_semantics='first raw predicted action of each response, not the executed action',
        missing_response_request_ids=missing,
        missing_action_semantics='zero placeholders only where policy.response_valid=false; exclude from analysis/training',
        timeline='one row per request at nominal FPS; source.* preserves original irregular clocks without resampling',
        image_semantics=manifest.get('image_scope'), summary_present=(source / 'summary.json').exists())
    (provenance / 'export.json').write_text(json.dumps(notes, indent=2) + '\n')
    (output / 'README.md').write_text('# Rollout review dataset\n\n'
        'One row per inference request. Nominal FPS is only the playback timeline; '
        'source.relative_time_s and source.wall_unix_ns retain original timing.\n\n'
        'action is the first raw prediction, not the executed action. policy.action_chunk '
        f'contains the complete flattened {shape} prediction. Missing responses use zero '
        'storage placeholders with policy.response_valid=false. These are not measured actions. '
        'Do not use this export directly as training demonstrations.\n\n'
        'source/events.jsonl retains recorded control attempts, measured joints and submitted commands. '
        'Images preserve the recorded resolution; no original camera stream is reconstructed.\n')
    print(json.dumps(notes, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo-id', default='local/g01-rollout-review')
    p.add_argument('--fps', type=int, default=3)
    p.add_argument('--streaming-encoding', action='store_true',
                   help='Encode video while adding frames instead of writing temporary PNGs')
    a = p.parse_args()
    if a.fps <= 0:
        p.error('--fps must be positive')
    export(a.source, a.output, a.repo_id, a.fps, streaming_encoding=a.streaming_encoding)
