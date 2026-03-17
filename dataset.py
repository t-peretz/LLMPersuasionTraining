"""Dataset wrappers and GSM8K preprocessing."""

import re
from typing import Any, Dict, List

from torch.utils.data import Dataset
from datasets import load_dataset

from prompt_parsing import build_A_prompt


class ListDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def load_gsm8k() -> List[Dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"]

    rows = []
    for ex in train:
        m = re.search(r"####\s*([-\d\.]+)", ex["answer"])
        if not m:
            continue
        q = ex["question"]
        rows.append(
            {
                "prompt": build_A_prompt(q),
                "question": q,
                "solution": m.group(1),
            }
        )
    return rows