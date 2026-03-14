from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch

from datasets.tokenizers.tokenizer_loader import get_tokenizer


class GSM8KDataset:

    def __init__(self, path="datasets/raw/gsm8k", seq_len=128):

        self.dataset = load_from_disk(path)

        self.tokenizer = get_tokenizer()

        self.seq_len = seq_len

    def tokenize(self, example):

        text = example["question"] + " " + example["answer"]

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.seq_len
        )

        return {
            "input_ids": torch.tensor(tokens["input_ids"])
        }

    def get_dataloader(self, batch_size=8):

        dataset = self.dataset["train"].map(self.tokenize)

        dataset.set_format(type="torch", columns=["input_ids"])

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return loader
