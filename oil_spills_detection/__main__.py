import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from PIL import Image
import numpy as np
import flet as ft
from flet import FilePicker, FilePickerResultEvent, Image as FletImage

# === Константы ===
class_names = ['background', 'oil', 'others', 'water']
class_colors = {
    'oil': (255, 0, 124),
    'others': (255, 204, 51),
    'water': (51, 221, 255)
}

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


# === Инициализация модели ===
def load_model(weights_path: str, num_classes: int = 4):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = maskrcnn_resnet50_fpn(weights=None)

    # Настройка box_predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Настройка mask_predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    # Загрузка весов
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device

# === Преобразование + инференс ===
def predict_image(model, device, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    output = outputs[0]
    masks = output['masks']
    labels = output['labels']
    scores = output['scores']

    np_image = np.array(image)
    result_image = np_image.copy()

    for mask, label, score in zip(masks, labels, scores):
        if score > 0.5:
            label = int(label)
            color = class_colors[class_names[label]]

            mask_np = mask[0].cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)

            # Наложение маски
            for c in range(3):
                result_image[:, :, c] = np.where(mask_bin == 1,
                                                 result_image[:, :, c] * 0.6 + color[c] * 0.4,
                                                 result_image[:, :, c])

    # Возвращаем результат
    return Image.fromarray(result_image.astype(np.uint8))

# === GUI на Flet ===
def main(page: ft.Page):
    page.title = "Oil Spill Segmentation (Mask R-CNN)"
    page.scroll = ft.ScrollMode.AUTO
    page.window_width = 800
    page.window_height = 600
    page.title = "Oil Spill Detection Viewer"
    page.scroll = "auto"

    # Легенда
    legend_items = []
    for name in class_names[1:]:  # Пропускаем 'background'
        color_hex = rgb_to_hex(class_colors[name])
        legend_items.append(
            ft.Row(
                controls=[
                    ft.Container(width=20, height=20, bgcolor=color_hex, border_radius=4),
                    ft.Text(name.capitalize())
                ],
                spacing=10,
                alignment=ft.MainAxisAlignment.START
            )
        )

    legend = ft.Column(
        controls=[
            ft.Text("Легенда классов:", size=16, weight=ft.FontWeight.BOLD)
        ] + legend_items,
        spacing=5,
        alignment=ft.MainAxisAlignment.START
    )

    page.add(legend)
    image_display = FletImage(src="", fit=ft.ImageFit.CONTAIN, expand=True)
    status_text = ft.Text("Выберите изображение", size=16)

    def on_result(e: FilePickerResultEvent):
        if e.files:
            path = e.files[0].path
            status_text.value = f"Инференс на: {os.path.basename(path)}..."
            page.update()

            # Прогон изображения через модель
            result = predict_image(model, device, path)

            # Сохраняем временный результат
            temp_path = "/tmp/predicted_result.png"
            result.save(temp_path)

            # Отображаем
            image_display.src = temp_path
            status_text.value = f"Готово: {os.path.basename(path)}"
            page.update()

    file_picker = FilePicker(on_result=on_result)
    pick_btn = ft.ElevatedButton("Выбрать изображение", on_click=lambda _: file_picker.pick_files(
        allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE))

    page.overlay.append(file_picker)
    page.add(
        status_text,
        pick_btn,
        ft.Container(image_display, padding=10, expand=True),
    )

# === Загрузка модели ===
model_path = "./mask_rcnn_oilspill_trained.pth"
model, device = load_model(model_path)

# === Запуск GUI ===
if __name__ == "__main__":
    ft.app(target=main)
