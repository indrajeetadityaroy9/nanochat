"""
The data root, <base_dir>/data, and the one way files get into it: fetch(path, download) calls download(path)
only if path is missing, holding a lock on path, so ranks that ask for the same file download it once.
download must put path in place only when it is complete.

HuggingFace dataset repos are read at pinned commits; their files live at <data_dir>/<org>/<repo>/<repo-relative path>.
"""

import os
import json

from filelock import FileLock
from huggingface_hub import hf_hub_download

from nanochat.common import get_base_dir


def get_data_dir():
    return os.path.join(get_base_dir(), "data")


def fetch(path, download):
    """path, after download(path) has created it if it was missing; concurrent callers wait for the first one."""
    with FileLock(path + ".lock"):
        if not os.path.exists(path):
            download(path)
    return path


def write_json(path, obj):
    """Write JSON atomically: readers see the old file or the complete new one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def fetch_repo_file(repo, revision, filename):
    """Local path of one file of a HuggingFace dataset repo at a commit, downloaded on first use."""
    local_dir = os.path.join(get_data_dir(), repo)
    return fetch(os.path.join(local_dir, filename),
                 lambda path: hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=local_dir))
