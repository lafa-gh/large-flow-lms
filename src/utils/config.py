"""Base config mixin: JSON save/load + HfArgumentParser-compatible config-first parsing."""

import json
import sys
import os
import dataclasses
from pathlib import Path

from transformers import HfArgumentParser, TrainingArguments


class BaseConfig:
    """Mixin for dataclass configs: adds save/load and JSON-first CLI parsing."""

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            return cls(**json.load(f))


def parse_config_and_training_args(config_cls, argv=None) -> tuple:
    """Parse config JSON (if present) + TrainingArguments with CLI overrides.

    Usage:
        config, training_args = parse_config_and_training_args(FlowConfig)

    Supports three modes:
        1. JSON only:    python train.py config.json
        2. CLI only:     python train.py --model_name_or_path ... --output_dir ...
        3. JSON + CLI:   python train.py config.json --learning_rate 1e-4  (CLI overrides JSON)
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = HfArgumentParser((config_cls, TrainingArguments))
    config_file = argv[0] if argv and argv[0].endswith(".json") else None

    if config_file:
        config_args = parser.parse_json_file(
            json_file=os.path.abspath(config_file), allow_extra_keys=False
        )

        other_args = argv[1:]
        arg_names = {
            arg.lstrip("-").split("=")[0] for arg in other_args if arg.startswith("--")
        }

        if arg_names:
            required_args = [
                (act.option_strings[0], "dummy")
                for act in parser._actions
                if act.required
                and not any(
                    act_s.lstrip("-") in arg_names for act_s in act.option_strings
                )
            ]
            required_args = [a for pair in required_args for a in pair]
            cli_args = parser.parse_args_into_dataclasses(
                args=other_args + required_args, look_for_args_file=False
            )
            all_args = []
            for cfg_dc, cli_dc in zip(config_args, cli_args):
                cli_d = {
                    k: v
                    for k, v in dataclasses.asdict(cli_dc).items()
                    if k in arg_names or f"no_{k}" in arg_names
                }
                all_args.append(dataclasses.replace(cfg_dc, **cli_d))
            config, training_args = all_args
        else:
            config, training_args = config_args
    else:
        config, training_args = parser.parse_args_into_dataclasses(
            args=argv, look_for_args_file=False
        )

    explicit = set()
    if config_file:
        with open(config_file) as handle:
            explicit.update(json.load(handle))
    explicit.update(arg.lstrip('-').split('=')[0] for arg in argv if arg.startswith('--'))
    config._explicit_fields = explicit
    return config, training_args
