"""
What sets the code corpora apart from the text ones, whose Hub files are used as they are: fetch materializes each code
source into zstd parquet files of one row per source file (swh.py, stack_v3.py), with the columns every code file has
(text, language, document_id, repository, path, license_type, detected_licenses), then its source's own metadata. Stack
v3 keeps only its source code files (stack_v3.py); Stack-Edu's 15 languages are its publisher's selection.
"""
