"""Resolve local checkpoints and Hub snapshots through one cache-aware path."""
from pathlib import Path
from huggingface_hub import snapshot_download


def resolve_checkpoint(source, *, revision=None, subfolder='', local_files_only=False):
    relative = Path(subfolder)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('subfolder must be a relative path within the model repo')
    local = Path(source)
    if local.exists():
        result = local / relative
        if not result.is_dir():
            raise FileNotFoundError(result)
        return str(result)
    prefix = subfolder.strip('/')
    patterns = [f'{prefix}/*'] if prefix else ['*']
    # Reading EMA/main should not download the raw weights a second time.
    root = snapshot_download(
        repo_id=source, revision=revision, local_files_only=local_files_only,
        allow_patterns=patterns, ignore_patterns=['raw/*'] if not prefix else None)
    result = Path(root) / relative
    if not result.is_dir():
        raise FileNotFoundError(f'Subfolder {subfolder!r} not found in {source}')
    return str(result)
