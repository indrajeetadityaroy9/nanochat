"""
Storage layer for every dataset: the local data root and pinned HuggingFace dataset repos.

Every download is locked per file and published with an atomic rename, so any number of ranks
and DataLoader workers on a node can ask for the same file: it is fetched exactly once and no
reader ever sees a partial file. Files already present are never re-checked against the network,
so a fully prepared data root also works offline (HF_HUB_OFFLINE=1) and mounted read-only.
"""

import os
import json
import shutil
import hashlib

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download

from nanochat.common import get_base_dir


def get_data_dir():
    """Root of all datasets: $NANOCHAT_DATA_DIR, default <base_dir>/data. Put it on fast local NVMe."""
    data_dir = os.environ.get("NANOCHAT_DATA_DIR") or os.path.join(get_base_dir(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def file_lock(path, timeout=-1):
    """Cross-process, cross-thread lock for one local path (lock files live in <data_dir>/.locks)."""
    lock_dir = os.path.join(get_data_dir(), ".locks")
    os.makedirs(lock_dir, exist_ok=True)
    name = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()
    return FileLock(os.path.join(lock_dir, name + ".lock"), timeout=timeout)


def fetch_once(local_path, download):
    """Ensure local_path (a file or directory) exists, calling download(tmp_path) at most once across processes on this node."""
    if os.path.exists(local_path):
        return local_path
    with file_lock(local_path):
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            tmp_path = f"{local_path}.tmp{os.getpid()}"
            try:
                download(tmp_path)
                os.replace(tmp_path, local_path)
            finally:
                if os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path)
                elif os.path.exists(tmp_path):
                    os.remove(tmp_path)
    return local_path


# -----------------------------------------------------------------------------
# HuggingFace dataset repos, always at a pinned revision

def list_repo_files(repo, revision, cache_dir):
    """All file paths of a HuggingFace dataset repo at a pinned revision; the listing is cached in cache_dir."""
    listing = os.path.join(cache_dir, f".files-{revision.replace('/', '--')}.json")
    def download(tmp_path):
        with open(tmp_path, "w") as f:
            json.dump(HfApi().list_repo_files(repo, repo_type="dataset", revision=revision), f)
    with open(fetch_once(listing, download)) as f:
        return json.load(f)


def fetch_repo_file(repo, revision, filename, local_dir):
    """Local path of one file of a HuggingFace dataset repo at a pinned revision, downloaded once if missing."""
    path = os.path.join(local_dir, filename)
    if not os.path.exists(path):
        # our lock, not only huggingface_hub's: a second concurrent hf_hub_download of the same file
        # deletes and re-downloads the file the first one just finished
        with file_lock(path):
            if not os.path.exists(path):
                hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=local_dir)
    return path
