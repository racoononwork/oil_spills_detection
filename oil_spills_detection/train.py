
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from dataset import OilSpillDataset
from model import get_model_instance
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = OilSpillDataset(DATA_PATH / "train/images", DATA_PATH / "train/masks", transform=get_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model_instance()
    model.to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} completed. Total loss: {total_loss:.4f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict()
    }, MODEL_SAVE_PATH)

    print("Training completed and model saved.")


if __name__ == "__main__":
    train()
