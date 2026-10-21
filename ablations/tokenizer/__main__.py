"""
The tokenizer experiment's steps (experiment.py): python -m ablations.tokenizer vocabularies | data | runs.
"""
import argparse

from ablations.tokenizer.experiment import data, runs, vocabularies

parser = argparse.ArgumentParser(prog="python -m ablations.tokenizer",
                                 description="The tokenizer ablation: its vocabularies, its data, its runs")
parser.add_argument("step", choices=("vocabularies", "data", "runs"))
{"vocabularies": vocabularies, "data": data, "runs": runs}[parser.parse_args().step]()
