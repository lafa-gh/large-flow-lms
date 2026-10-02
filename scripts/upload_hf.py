"""Upload a locally verified package to an existing private model repository."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',required=True);p.add_argument('--repo-id',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.folder)
    manifest=json.loads((root/'export_manifest.json').read_text())
    if manifest['repo_id']!=a.repo_id:raise ValueError('Manifest repository mismatch')
    for name,expected in manifest['files'].items():
        f=root/name
        if not f.is_file() or digest(f)!=expected['sha256']:
            raise ValueError(f'Package changed after export: {name}')
    api=HfApi();info=api.model_info(a.repo_id)
    if not info.private:raise ValueError('This uploader is for the private staging repositories')
    commit=api.upload_folder(repo_id=a.repo_id,repo_type='model',folder_path=str(root),
        parent_commit=info.sha,commit_message='Add Lafa Transformers model checkpoint',
        ignore_patterns=['__pycache__/*','*.pyc','.cache/*'])
    report={'repo_id':a.repo_id,'commit':commit.oid,'url':commit.commit_url,'private':True}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2));print(report)


if __name__=='__main__':main()
