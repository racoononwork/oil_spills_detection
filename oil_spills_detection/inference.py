import torch
from model import get_model_instance
from config import MODEL_SAVE_PATH, NUM_CLASSES
from pathlib import Path


def load_model(model_path: Path = MODEL_SAVE_PATH) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model_instance(num_classes=NUM_CLASSES, weights=False)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = load_model()
    print("Model loaded and ready for inference.")
