"""Checkpoint directory discovery."""

from pathlib import Path


def latest_checkpoint(output_dir: str) -> str | None:
    """Find the highest numbered checkpoint directory, if any."""
    latest = None
    latest_step = -1
    for path in Path(output_dir).glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if path.is_dir() and step.isascii() and step.isdigit():
            number = int(step)
            if number > latest_step:
                latest, latest_step = path, number
    return str(latest) if latest is not None else None
