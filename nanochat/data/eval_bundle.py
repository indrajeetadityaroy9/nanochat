"""
CORE benchmark data (the DCLM eval bundle), fetched once into <data_dir>/eval_bundle.
"""

import os
import shutil
import zipfile
import tempfile
import urllib.request

from nanochat.data.storage import fetch, get_data_dir

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def get_eval_bundle_dir():
    """Directory holding core.yaml, eval_meta_data.csv and eval_data/, downloaded and extracted once per node."""
    def download(path):
        with tempfile.TemporaryDirectory(dir=get_data_dir()) as scratch:
            zip_path = os.path.join(scratch, "eval_bundle.zip")
            print(f"Downloading {EVAL_BUNDLE_URL}...")
            urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(scratch)
            shutil.move(os.path.join(scratch, "eval_bundle"), path)
    return fetch(os.path.join(get_data_dir(), "eval_bundle"), download)
