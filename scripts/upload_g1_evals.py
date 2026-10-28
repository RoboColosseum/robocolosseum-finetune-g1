"""Create-only private HF uploads of verified minimal G1 LeRobot datasets."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    from huggingface_hub import HfApi,hf_hub_download
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    args=parser.parse_args()
    api=HfApi(token=args.token_file.read_text().strip())
    identity=api.whoami()
    assert 'RoboColosseum' in [o.get('name') for o in identity.get('orgs',[])]
    args.audit.mkdir(exist_ok=True,parents=True)
    ledger_path=args.audit/'hf-upload-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    selection=json.loads((args.audit/'selection.json').read_text())
    names=sorted(selection['groups'])
    assert len(names)==26 and sum(map(len,selection['groups'].values()))==653
    while True:
        progress=False
        for name in names:
            if ledger.get(name,{}).get('state')=='verified':continue
            local_report=args.audit/(name+'-verification.json')
            if not local_report.exists():continue
            local=json.loads(local_report.read_text())
            root=args.root/name
            manifest=json.loads((args.audit/(name+'-files.json')).read_text())
            assert {p.name for p in root.iterdir()}=={'meta','data','videos'}
            assert all(p.split('/')[0] in ('meta','data','videos') for p in manifest)
            repo_id='RoboColosseum/'+name
            if name not in ledger:
                # A collision is never treated as authorization to overwrite existing data.
                url=api.create_repo(repo_id,repo_type='dataset',private=True,exist_ok=False)
                ledger[name]=dict(state='created',repo_id=repo_id,url=str(url),private=True)
                write_json(ledger_path,ledger)
            entry=ledger[name]
            info=api.dataset_info(repo_id,files_metadata=True)
            assert info.private is True
            for sibling in info.siblings:
                if sibling.rfilename=='.gitattributes':continue
                assert sibling.rfilename in manifest,'Existing unexpected remote file: '+sibling.rfilename
                expected=manifest[sibling.rfilename]
                assert sibling.size==expected['size']
                if sibling.lfs is not None:
                    assert sibling.lfs.sha256==expected['sha256']
                else:
                    raw=(root/sibling.rfilename).read_bytes()
                    assert sibling.blob_id==hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            write_json(args.audit/'hf-status.json',dict(state='uploading',repo_id=repo_id,verified=sum(v['state']=='verified' for v in ledger.values()),total=26))
            commit=api.upload_folder(repo_id=repo_id,repo_type='dataset',folder_path=root,
                allow_patterns=['meta/**','data/**','videos/**'],
                commit_message=f'Add {local["episodes"]} G1 evaluation episodes in LeRobot v3 format',
                parent_commit=info.sha)
            entry.update(state='uploaded',commit=commit.oid,episodes=local['episodes'],frames=local['frames'],bytes=local['bytes'])
            write_json(ledger_path,ledger)
            remote=api.dataset_info(repo_id,revision=commit.oid,files_metadata=True)
            assert remote.private is True
            files={f.rfilename:f for f in remote.siblings if f.rfilename!='.gitattributes'}
            assert set(files)==set(manifest),(repo_id,'File inventory mismatch')
            for path,expected in manifest.items():
                item=files[path]
                assert item.size==expected['size'],(repo_id,path,'size')
                if item.lfs is not None:
                    assert item.lfs.sha256==expected['sha256'],(repo_id,path,'sha256')
                else:
                    raw=(root/path).read_bytes()
                    expected_blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                    assert item.blob_id==expected_blob,(repo_id,path,'git blob')
            for path in ('meta/info.json','meta/tasks.parquet'):
                downloaded=Path(hf_hub_download(repo_id,filename=path,repo_type='dataset',revision=commit.oid,
                    token=args.token_file.read_text().strip(),cache_dir=args.audit/'hf-readback-cache'))
                assert sha(downloaded)==manifest[path]['sha256']
            entry.update(state='verified',files=len(files),verification='exact inventory, all sizes, LFS SHA-256 / Git blob hashes, metadata download readback')
            write_json(ledger_path,ledger)
            print(json.dumps(entry),flush=True)
            progress=True
        verified=sum(v['state']=='verified' for v in ledger.values())
        write_json(args.audit/'hf-status.json',dict(state='complete' if verified==26 else 'waiting_for_datasets',verified=verified,total=26,
            episodes=sum(v.get('episodes',0) for v in ledger.values() if v['state']=='verified')))
        if verified==26:
            assert sum(v['episodes'] for v in ledger.values())==653
            assert sum(v['frames'] for v in ledger.values())==99725
            break
        if not progress:time.sleep(15)


if __name__=='__main__':main()
