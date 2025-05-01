from torch.utils.data import DataLoader
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, ShiftScaleRotate, Normalize
from albumentations.pytorch import ToTensorV2
from dataset import OilSpillDataset
from model import OilSpillModel
from trainer import Trainer


def get_transforms() -> Compose:
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def main() -> None:
    data_path = "/content/drive/MyDrive/Sadekov_task/dataset/"
    train_images = f"{data_path}/train/images"
    train_masks = f"{data_path}/train/masks"

    transform = get_transforms()
    dataset = OilSpillDataset(train_images, train_masks, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model_wrapper = OilSpillModel(num_classes=4)
    trainer = Trainer(model_wrapper.get_model(), model_wrapper.get_device())
    trainer.train(dataloader, epochs=10)


if __name__ == "__main__":
    main()