"""
Optional NeMo Curator stage: quality-filter a registered corpus into a curated local corpus '<dataset>-curated'.

The registered corpora are already filtered by their publishers; this stage is for corpora that are not,
or to hold data to a stricter bar. It runs NeMo Curator's CPU heuristic quality filters (document length,
non-alphanumeric and symbol ratios, sentence punctuation, n-gram repetition) over the corpus' raw parquet
files and writes the surviving documents as parquet to $NANOCHAT_DATA_DIR/raw/<dataset>-curated/{train,val}/.
Everything downstream treats '<dataset>-curated' as a corpus of its own:
    python -m nanochat.tokenizer --dataset=<dataset>-curated
    python -m nanochat.data.compile --dataset=<dataset>-curated

NeMo Curator needs Python >= 3.11 and brings Ray, so this runs in the NeMo Curator container
(nvcr.io/nvidia/nemo-curator, the `curate` service of docker/compose.yaml), not in the training environment.

python -m nanochat.data.curate --dataset=climbmix --max-files=170
"""

import os
import argparse

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic import NonAlphaNumericFilter, PunctuationFilter, SymbolsToWordsFilter, WordCountFilter
from nemo_curator.stages.text.filters.heuristic.repetition import RepeatingTopNGramsFilter
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

from nanochat.data.sources import CURATED, DATASETS, get_dataset, get_raw_dir, list_raw_files, fetch_raw_file


def quality_filter(text_field):
    """Gopher/C4-style heuristic document filters, cheapest first, applied in one pipeline stage."""
    filters = [
        WordCountFilter(min_words=50, max_words=100_000),
        NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
        SymbolsToWordsFilter(max_symbol_to_word_ratio=0.1),
        PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
        RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2),
        RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
        RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16),
    ]
    return ScoreFilter(filter_obj=filters, text_field=text_field)


def curate(dataset, max_files):
    text_field = get_dataset(dataset).text_column
    out_dir = get_raw_dir(dataset + CURATED)
    for split in ["train", "val"]:
        files = list_raw_files(dataset, split)
        if split == "train" and max_files > 0:
            files = files[:max_files]
        split_dir = os.path.join(out_dir, split)
        assert not os.path.exists(split_dir), f"{split_dir} exists: delete it to curate again"
        print(f"Curating {len(files)} {split} files of '{dataset}' into {split_dir}")
        stages = [
            ParquetReader(file_paths=[fetch_raw_file(dataset, f) for f in files], fields=[text_field]),
            quality_filter(text_field),
            ParquetWriter(path=split_dir),
        ]
        Pipeline(name=f"curate-{dataset}-{split}", description="nanochat heuristic quality filtering", stages=stages).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality-filter a registered corpus with NeMo Curator into '<dataset>-curated'")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS), help="registered corpus to curate")
    parser.add_argument("--max-files", type=int, default=-1, help="curate only the first N raw train files (-1 = all)")
    args = parser.parse_args()
    ray_client = RayClient()
    ray_client.start()
    try:
        curate(args.dataset, args.max_files)
    finally:
        ray_client.stop()
