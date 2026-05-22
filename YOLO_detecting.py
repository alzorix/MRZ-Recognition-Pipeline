import logging
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

DOWNLOAD_DIR = "downloaded_photos"
CROPS_DIR = "mrz_crops"

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
main_logger = logging.getLogger('AI_module')


def load_model(model_path: str = "mrz.pt") -> YOLO:
    return YOLO(model_path)


def detect_and_crop_mrz(model: YOLO, image_path: str, crops_dir: str):
    image = cv2.imread(image_path)
    if image is None:
        main_logger.warning(f"Не удалось загрузить: {image_path}")
        return None

    results = model(image, verbose=False, conf=0.3)

    obb = results[0].obb
    if obb is None or len(obb) == 0:
        main_logger.warning(f"MRZ не найдена: {image_path}")
        return None

    main_logger.debug(f"Найдено OBB боксов: {len(obb)}, conf: {obb.conf.tolist()}")

    # Берём все боксы (обе строки MRZ) и строим общий bbox
    x1 = int(obb.xyxy[:, 0].min().item())
    y1 = int(obb.xyxy[:, 1].min().item())
    x2 = int(obb.xyxy[:, 2].max().item())
    y2 = int(obb.xyxy[:, 3].max().item())

    # Отступ
    padding = 5
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = image[y1:y2, x1:x2]

    Path(crops_dir).mkdir(exist_ok=True)
    filename = Path(image_path).name
    crop_path = os.path.join(crops_dir, filename)
    cv2.imwrite(crop_path, crop)

    main_logger.debug(f"MRZ вырезана: {filename} - {crop_path}")
    return crop_path


def process_all(download_dir: str = DOWNLOAD_DIR, crops_dir: str = CROPS_DIR):
    model = load_model("mrz.pt")

    image_files = list(Path(download_dir).glob("*.*"))
    if not image_files:
        main_logger.critical("Нет изображений для обработки")
        return []

    crop_paths = []
    for image_path in image_files:
        crop_path = detect_and_crop_mrz(model, str(image_path), crops_dir)
        if crop_path:
            crop_paths.append(crop_path)

    main_logger.info(f"Итого: {len(crop_paths)}/{len(image_files)} MRZ успешно вырезано")
    return crop_paths


if __name__ == '__main__':
    process_all()