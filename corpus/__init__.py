"""
The pretraining corpus under paths.CORPUS, one module per step. Run order: prepare, curate, train_tokenizer, publish
(push; pull fetches the curated shards and the tokenizer), tokenize, pack; loader reads the packs.
"""
