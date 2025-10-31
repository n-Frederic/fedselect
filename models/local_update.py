"""
LocalUpdate migrated from federated-learning-master/models/Update.py
Performs local training and returns state_dict.
"""

from typing import Dict, Optional, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class LocalUpdate:
    """
    LocalUpdate encapsulates client-side training.
    Usage:
        local = LocalUpdate(device, dataset, idxs, args)
        w, loss = local.train(net=copy_of_global_model)
    """

    def __init__(self,
                 device: torch.device,
                 dataset: Dataset,
                 idxs: list,
                 batch_size: int = 32,
                 local_epochs: int = 1,
                 lr: float = 0.01,
                 momentum: float = 0.5,
                 verbose: bool = False):
        self.device = device
        self.dataset = dataset
        self.idxs = idxs
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.verbose = verbose

        # build dataloader for client's local indices
        self.trainloader = self._build_dataloader()

    def _build_dataloader(self) -> DataLoader:
        subset = torch.utils.data.Subset(self.dataset, self.idxs)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    def train(self, net: nn.Module, criterion: Optional[nn.Module] = None) -> Tuple[Dict, float]:
        net.train()
        net.to(self.device)
        criterion = criterion or nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum)

        epoch_loss = []
        for epoch in range(self.local_epochs):
            batch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            avg = batch_loss / (batch_idx + 1) if (batch_idx + 1) > 0 else 0.0
            epoch_loss.append(avg)
            if self.verbose:
                print(f"[Local] epoch {epoch+1}/{self.local_epochs} loss {avg:.4f}")
        # return local model parameters and average loss
        return net.cpu().state_dict(), float(sum(epoch_loss) / max(len(epoch_loss), 1))