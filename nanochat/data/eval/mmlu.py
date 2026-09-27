"""
The MMLU dataset (https://huggingface.co/datasets/cais/mmlu), read by pretraining decontamination (decontam.py), which
hashes every string of an item.
"""

from nanochat.data.task import Task, load_hub_dataset, render_mc

REPO = "cais/mmlu"
REVISION = "c30699e8356da336a370243923dbaf21066bb9fe" # pinned commit, files are <subset>/<split>-*.parquet

class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        assert split in ["auxiliary_train", "validation", "dev", "test"], f"split {split} must be auxiliary_train|validation|dev|test"
        self.ds = load_hub_dataset(REPO, REVISION, f"{subset}/{split}-*.parquet").shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"] # the text of each choice
        answer = row["answer"] # index of the answer, e.g. 0,1,2,3 (for A,B,C,D)
        subject = row["subject"] # e.g. "college_biology", "college_chemistry", etc.
        assert len(choices) == 4, "MMLU should have 4 choices"
        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }
        return conversation
