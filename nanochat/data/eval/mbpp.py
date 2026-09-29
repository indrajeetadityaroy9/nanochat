"""
MBPP, Mostly Basic Python Problems (https://huggingface.co/datasets/google-research-datasets/mbpp, the full
configuration): a task description, a reference solution and its assert tests per problem.
"""

from nanochat.data.task import Task, load_hub_dataset

REPO = "google-research-datasets/mbpp"
REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c" # pinned commit; splits full/{train,validation,test,prompt}-*.parquet

class MBPP(Task):

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_hub_dataset(REPO, REVISION, f"full/{split}-*.parquet")

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ A problem: task_id, text (the description), code (the reference solution), test_list (assert statements). """
        return self.ds[index]
