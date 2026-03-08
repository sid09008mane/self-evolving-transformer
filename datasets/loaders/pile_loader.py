from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch

from datasets.tokenizers.tokenizer_loader import get_tokenizer


class PileDataset:

    def __init__(self, path="datasets/raw/pile", seq_len=128):

        self.dataset = load_from_disk(path)

        self.tokenizer = get_tokenizer()

        self.seq_len = seq_len

    def tokenize(self, example):

        tokens = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=self.seq_len
        )

        return {
            "input_ids": torch.tensor(tokens["input_ids"])
        }

    def get_dataloader(self, batch_size=8):

        dataset = self.dataset.map(self.tokenize)

        dataset.set_format(type="torch", columns=["input_ids"])

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return loader
