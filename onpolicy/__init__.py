"""The on-policy post-SFT stage (docs/post-training.md §8): rollouts of the SFT model on KodCode prompts, verified by
KodCode's tests; rejection-sampling SFT on its shortest terminating successes, then DPO of those against its own
reasoning loops."""
