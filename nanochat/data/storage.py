"""
The data root, <base_dir>/data, the one-download-per-node fetch for files there, and the JSON writer for the manifests
and statistics kept under it.
"""

import os
import json

from filelock import FileLock

from nanochat.common import get_base_dir


def get_data_dir():
    return os.path.join(get_base_dir(), "data")


def fetch(path, download):
    """path, after download(path) has created it if it was missing; concurrent callers wait for the first one. The check
    runs under the lock: huggingface_hub's own lock (hf_hub_download with local_dir, 2.0.0) checks before locking, so every
    concurrent caller downloads again and first deletes the file an earlier one just completed."""
    with FileLock(path + ".lock"):
        if not os.path.exists(path):
            download(path)
    return path


def write_json(path, obj):
    """Write JSON atomically and durably: readers see the old file or the complete new one, also after a power loss
    (the data reaches the disk before the rename, as torch.distributed.checkpoint's sync_files does)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
