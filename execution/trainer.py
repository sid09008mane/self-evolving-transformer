import torch
import torch.nn as nn
from torch.optim import Adam

from datasets.loaders.dataset_router import DatasetRouter
from execution.model_builder import ModelBuilder


class Trainer:

    def __init__(
        self,
        device=None,
        batch_size=8,
        seq_len=128,
        lr=3e-4,
        steps=50
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.steps = steps

        self.router = DatasetRouter()

        self.builder = ModelBuilder()

        self.loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------
    # Train a single architecture
    # ------------------------------------------------

    def train_genome(self, genome, dataset_name="openwebtext"):

        model = self.builder.build(genome)

        model = model.to(self.device)

        optimizer = Adam(model.parameters(), lr=self.lr)

        loader = self.router.get_dataloader(
            dataset_name,
            batch_size=self.batch_size
        )

        model.train()

        total_loss = 0
        step = 0

        for batch in loader:

            input_ids = batch["input_ids"].to(self.device)

            optimizer.zero_grad()

            logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            step += 1

            if step >= self.steps:
                break

        avg_loss = total_loss / step

        # Convert loss → fitness
        fitness = 1 / (avg_loss + 1e-6)

        return fitness
