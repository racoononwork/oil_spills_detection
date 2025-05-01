from pathlib import Path

DATA_PATH: Path = Path("/home/ristle/PycharmProjects/PythonProject/dataset")
MODEL_SAVE_PATH: Path = Path("/home/ristle/PycharmProjects/PythonProject/mask_rcnn_oilspill_trained.pth")
NUM_CLASSES: int = 4  # 3 classes + background
EPOCHS: int = 10
BATCH_SIZE: int = 2
LEARNING_RATE: float = 1e-4
