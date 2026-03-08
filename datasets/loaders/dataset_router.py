from datasets.loaders.openwebtext_loader import OpenWebTextDataset
from datasets.loaders.pile_loader import PileDataset
from datasets.loaders.gsm8k_loader import GSM8KDataset


class DatasetRouter:

    def __init__(self):

        self.datasets = {
            "openwebtext": OpenWebTextDataset(),
            "pile": PileDataset(),
            "gsm8k": GSM8KDataset()
        }

    def get_dataloader(self, name, batch_size=8):

        dataset = self.datasets[name]

        return dataset.get_dataloader(batch_size=batch_size)
