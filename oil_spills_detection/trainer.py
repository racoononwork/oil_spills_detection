from typing import Any
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam


class Trainer:
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def train(self, dataloader: DataLoader, epochs: int = 10) -> None:
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for images, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")