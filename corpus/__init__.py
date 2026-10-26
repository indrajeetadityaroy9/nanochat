"""
The pretraining corpus under paths.CORPUS, one module per step. Run order: prepare, curate, publish, the tokenizer
(ablations/tokenizer_split_merge.py trains the arms; sa-bpe's copied to CORPUS/tokenizer), tokenize, pack; loader
reads the packs.
"""
