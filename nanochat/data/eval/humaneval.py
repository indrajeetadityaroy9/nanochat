"""
HumanEval coding problems (https://huggingface.co/datasets/openai/openai_humaneval), read by pretraining
decontamination (decontam.py), which hashes every string of an item: prompt, canonical solution and tests.
"""

from nanochat.data.task import Task, load_hub_dataset

REPO = "openai/openai_humaneval"
REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544" # pinned commit, the only split is openai_humaneval/test-*.parquet

class HumanEval(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_hub_dataset(REPO, REVISION, "openai_humaneval/test-*.parquet").shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        prompt = row['prompt'] # prompts in HumanEval are the beginning of the program
        solution = row['canonical_solution'] # the correct continuation of the program
        entry_point = row['entry_point'] # the function to check
        test = row['test'] # the test cases
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point,
            "test": test,
        }
        return conversation
