"""
The on-disk roots of a nanochat base ($NANOCHAT_BASE_DIR: tokenizer, data, checkpoints, metrics, caches) and the JSON
writer for the files kept under them.
"""

import json
import os


def base_dir():
    return os.environ["NANOCHAT_BASE_DIR"]


def data_dir():
    """The data root, <base_dir>/data (layout: nanochat/data/__init__.py)."""
    return os.path.join(base_dir(), "data")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)
