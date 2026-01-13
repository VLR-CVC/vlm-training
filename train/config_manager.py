from typing import Any

from dataclasses import fields, is_dataclass

import tyro
import tomllib


class ConfigManager:
    def __init__(self, config_cls):
        self.config_cls = config_cls

    def parse_args(self, args):
        toml_contents = self._maybe_load_toml(args)

        base_config = (
            self._dict_to_dataclass(self.config_cls, toml_contents)
            if toml_contents
            else self.config_cls()
        )

        self.config = tyro.cli(
            self.config_cls, args=args, default=base_config,
        )

        return self.config

    def _maybe_load_toml(self, args: list[str]) -> dict[str, Any] | None:
        valid_keys = {"--config"}
        for i, arg in enumerate(args):
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in valid_keys:
                    file_path = value
                    break
            elif i < len(args) - 1 and arg in valid_keys:
                file_path = args[i + 1]
                break
        else:
            return None

        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
            logger.exception(f"Error while loading config file: {file_path}")
            raise e

    def _dict_to_dataclass(self, cls, data: dict[str, Any]) -> Any:
        """Convert dictionary to dataclass, handling nested structures."""
        if not is_dataclass(cls):
            return data

        valid_fields = set(f.name for f in fields(cls))
        if invalid_fields := set(data) - valid_fields:
            raise ValueError(
                f"Invalid field names in {cls} data: {invalid_fields}.\n"
                "Please modify your .toml config file or override these fields from the command line.\n"
                "Run `NGPU=1 ./run_train.sh --help` to read all valid fields."
            )

        result = {}
        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                if is_dataclass(f.type) and isinstance(value, dict):
                    result[f.name] = self._dict_to_dataclass(f.type, value)
                else:
                    result[f.name] = value
        return cls(**result)
