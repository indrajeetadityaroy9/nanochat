"""
The training corpora under paths.CORPUS: the pretraining corpus's own steps in pretrain/, the SFT corpus's in
posttrain/, the steps and types both use here. Pretraining run order: pretrain.prepare, pretrain.curate,
decontaminate, train_tokenizer, pretrain.publish (push; pull fetches the curated shards, their decontamination record
and the tokenizer), tokenize, pack; then posttrain.build for the SFT corpus; loader reads the packs.
"""
