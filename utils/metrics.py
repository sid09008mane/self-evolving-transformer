import torch

def accuracy(preds, labels):

    preds = torch.argmax(preds, dim=-1)

    correct = (preds == labels).sum().item()

    return correct / len(labels)
