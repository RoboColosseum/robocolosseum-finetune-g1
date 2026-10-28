"""Audit recorded files and create task/model named links without moving originals."""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_deploy.recording import rollout_identity


def main():
    root = ROOT / 'outputs/rollouts'
    named = root / 'by-name'
    named.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [n for n in dirs if not (Path(folder)/n).is_symlink() and (Path(folder)/n) != named]
        if 'manifest.json' not in files:
            continue
        run = Path(folder)
        manifest = json.loads((run/'manifest.json').read_text())
        task, model = rollout_identity(manifest.get('config', {}), manifest.get('policy_schema'))
        prefix = f'{task}__{model}__'
        alias = named / (run.name if run.name.startswith(prefix) else prefix + run.name)
        if alias.is_symlink():
            if alias.resolve() != run.resolve():
                raise RuntimeError(f'Link conflict: {alias}')
        elif alias.exists():
            raise RuntimeError(f'Path conflict: {alias}')
        else:
            alias.symlink_to(os.path.relpath(run, named), target_is_directory=True)
        summary = json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {}
        counts = Counter(); missing_payloads = invalid_lines = 0
        requests = set(); responses = set()
        if (run/'events.jsonl').exists():
            with (run/'events.jsonl').open() as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        invalid_lines += 1; continue
                    counts[event.get('event', 'unknown')] += 1
                    if 'payload_file' in event and not (run/event['payload_file']).is_file():
                        missing_payloads += 1
                    if event.get('event') == 'policy_request': requests.add(event['request_id'])
                    if event.get('event') == 'policy_response': responses.add(event['request_id'])
        rows.append(dict(task=task, model=model, mode=manifest.get('mode'),
                         flush_complete=bool(summary.get('flush_complete')), summary_present=bool(summary),
                         dropped=sum(summary.get('dropped', {}).values()) if summary else None, writer_error=summary.get('writer_error'),
                         missing_payloads=missing_payloads, invalid_event_lines=invalid_lines,
                         unmatched_requests=len(requests-responses), observations=counts['observation'],
                         policy_requests=counts['policy_request'], policy_responses=counts['policy_response'],
                         commands_submitted=counts['command_submitted'], dry_run_commands=counts['dry_run_command'],
                         original_path=str(run), named_path=str(alias)))
    rows.sort(key=lambda x:x['original_path'])
    report = dict(runs=len(rows), complete_flush=sum(r['flush_complete'] for r in rows),
                  missing_summary=sum(not r['summary_present'] for r in rows),
                  runs_with_dropped_events=sum((r['dropped'] or 0)>0 for r in rows),
                  missing_payloads=sum(r['missing_payloads'] for r in rows),
                  invalid_event_lines=sum(r['invalid_event_lines'] for r in rows),
                  runs_with_recorded_inputs=sum(r['observations']+r['policy_requests']>0 for r in rows),
                  runs_with_submitted_commands=sum(r['commands_submitted']>0 for r in rows))
    (root/'index.json').write_text(json.dumps({'summary':report,'runs':rows},indent=2)+'\n')
    if rows:
        with (root/'index.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(report,indent=2))
    print('INCOMPLETE',json.dumps([{'path':r['original_path'],'summary_present':r['summary_present'],'dropped':r['dropped'],'writer_error':r['writer_error']} for r in rows if not r['flush_complete']],indent=2))


if __name__ == '__main__':
    main()
