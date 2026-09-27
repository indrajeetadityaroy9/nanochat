"""
What sets the code corpora apart from the general-domain ones, whose Hub files are used as they are:
- fetch materializes each code source into zstd parquet files of one row per source file (swh.py, stack_v3.py): the
  columns every code file has (text, language, document_id, repository, path, license_type, detected_licenses), then
  its source's own metadata;
- training reads RefineCode through SAMPLING (sources.read_training_documents);
- compile splits a file longer than a row losslessly at line ends (segment).
"""

import numpy as np

# Share of each RefineCode program_lang that OpenCoder trained on (arXiv 2411.04905, 2.1.1: Java 449 -> 200 GB, HTML
# 474 -> 64 GB); other languages are kept whole
SAMPLING = {"refinecode_stackv2_reconstructed": {"java": 200 / 449, "html": 64 / 474}}


def segment(doc, row_len, ends_line):
    """Pieces of at most row_len tokens that cover a BOS-prefixed document exactly, each starting with its BOS: a piece
    ends after the last token of its window that ends a line (ends_line, over the vocabulary), or with the window when
    none does."""
    if len(doc) <= row_len:
        return [doc]
    pieces, start = [], 1
    while len(doc) - start > row_len - 1:
        window = doc[start:start + row_len - 1]
        breaks = np.flatnonzero(ends_line[window])
        end = start + (breaks[-1] + 1 if breaks.size else len(window))
        pieces.append(np.concatenate((doc[:1], doc[start:end])))
        start = end
    pieces.append(np.concatenate((doc[:1], doc[start:])))
    return pieces
