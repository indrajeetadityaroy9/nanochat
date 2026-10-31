"""The on-policy post-SFT stage (docs/post-training.md §8): rollouts of the SFT model on its SFT train prompts, FTPO at
its reasoning loops' first repeats, then DPO on the FTPO model's KodCode rollouts verified by KodCode's tests: loop
pairs (a loop's repeat against a successful continuation from the same history) and outcome pairs (a success against
a failure of the same prompt)."""
