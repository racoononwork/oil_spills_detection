
## 🛢 Oil Spills Detection

Приложение для визуализации результатов модели `Mask R-CNN` по детекции разливов нефти, с графическим интерфейсом на [Flet](https://flet.dev). Запуск через `poetry`.

---

###  Установка и запуск

#### 1. Клонируйте репозиторий

```bash
git clone https://github.com/racoononwork/oil_spills_detection
cd oil_spills_detection
```

#### 2. Установите зависимости через [Poetry](https://python-poetry.org/)

```bash
poetry install
```

#### 3. Запуск проекта

```bash
poetry run python -m oil_spills_detection
```

---

### 🖥️ Описание

- 📷 Используется модель `maskrcnn_resnet50_fpn` из `torchvision` для предсказания масок.
- 🎨 Интерфейс построен с помощью `Flet` и отображает изображения с наложенными масками.
- 📊 Визуализируются предсказания на валидационной и тестовой выборках.
- 🏷️ Поддерживаются классы: `oil`, `others`, `water`.

---

### 📁 Структура проекта

```
oil_spills_detection/
├── __main__.py         # Точка входа (GUI на Flet)
├── model_loader.py     # Загрузка модели и весов
├── dataset.py          # Кастомный Dataset
├── inference.py        # Предсказание и визуализация
├── utils.py            # Утилиты (например, преобразование RGB в HEX)
├── ...
pyproject.toml
README.md
```

---

### 🧠 Требования

- Python ≥ 3.8
- torch
- torchvision
- flet
- matplotlib
- pillow
- tqdm

Все зависимости указаны в `pyproject.toml` и автоматически устанавливаются через Poetry.

---

### 📝 Пример запуска

```bash
poetry run python -m oil_spills_detection
```


