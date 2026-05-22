

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ПОДГОТОВКА ИЗОБРАЖЕНИЯ

def preprocess(img: np.ndarray) -> np.ndarray:
    # Переводим в чёрно-белое
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Если изображение слишком маленькое — увеличиваем..
    h, w = gray.shape
    if h < 100:
        scale = 150 / h
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # Убираем шум
    gray = cv2.fastNlMeansDenoising(gray, h=15)

    # Адаптивный порог: делаем текст чётким
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,  # размер окна анализа (должен быть нечётным)
        C=15  # насколько порог сдвигается от среднего
    )

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# РАСПОЗНАВАНИЕ ТЕКСТА

def run_ocr(img: np.ndarray) -> list[str]:
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,  # Отключаем классификацию ориентации документа
        use_doc_unwarping=False,              # Отключаем выпрямление документа
        use_textline_orientation=False,       # Отключаем определение ориентации строк
        lang='en',                            # или 'ru', 'en' и т.д. [citation:7]
        ocr_version="PP-OCRv5",                 # Явно указываем версию (необязательно, но наглядно) [citation:1]
        enable_mkldnn=False
    )

    lines = []
    for page in ocr.predict(img):
        if not page:
            continue
        texts = page.get("rec_texts", [])
        boxes = page.get("dt_polys", page.get("dt_boxes", []))
        # Сортируем блоки по Y-координате
        for _, text in sorted(zip(boxes, texts), key=lambda x: x[0][0][1]):
            if text:
                lines.append(text)
    return lines


# ОЧИСТКА СТРОК

# Символы MRZ
VALID = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")


def clean(line: str) -> str:
    line = line.upper().strip().replace(" ", "")
    # Всё что не MRZ-символ — заменяем на <
    return "".join(c if c in VALID else "<" for c in line)


# ПАРСИНГ
# MRZ паспорта (TD3): 2 строки по 44 символа.
# Каждый символ стоит на строго заданной позиции по стандарту ICAO 9303.
#
# Строка 1:  [0]   — тип документа (P = паспорт)
#            [2:5] — страна выдачи (JPN, RUS...)
#            [5:44]— имя: ФАМИЛИЯ<<ИМЯ<ОТЧЕСТВО (разделитель <<)
#
# Строка 2:  [0:9] — номер документа
#            [10:13]— гражданство
#            [13:19]— дата рождения YYMMDD
#            [20]  — пол (M/F)
#            [21:27]— срок действия YYMMDD
#            [28:42]— личный номер

def parse_date(yymmdd: str) -> str:
    """YYMMDD → YYYY-MM-DD"""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return yymmdd
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:]
    year = f"19{yy}" if int(yy) > 30 else f"20{yy}"
    return f"{year}-{mm}-{dd}"


def parse_mrz(lines: list[str]) -> dict:
    if len(lines) < 2:
        return {"error": "нужно минимум 2 строки"}

    l1 = lines[0].ljust(44, "<")[:44]
    l2 = lines[1].ljust(44, "<")[:44]

    name_parts = l1[5:44].split("<<", 1)

    return {
        "тип_документа": l1[0],
        "страна_выдачи": l1[2:5],
        "фамилия": name_parts[0].replace("<", " ").strip(),
        "имя": name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else "",
        "номер_документа": l2[0:9].replace("<", ""),
        "гражданство": l2[10:13],
        "дата_рождения": parse_date(l2[13:19]),
        "пол": l2[20],
        "срок_действия": parse_date(l2[21:27]),
        "Контрольные_цифры": l2[28:42].replace("<", ""),
    }


def recognize_mrz(path: str) -> dict:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не могу открыть: {path}")

    img = preprocess(img)  # 1. Подготовка
    lines = run_ocr(img)  # 2. OCR
    lines = [clean(l) for l in lines if len(l) >= 20]  # 3. Очистка
    fields = parse_mrz(lines)  # 4. Парсинг

    return {"строки": lines, "поля": fields}


if __name__ == "__main__":
    from YOLO_detecting import process_all
    results = process_all()
    for path in results:
        result = recognize_mrz(path)

        print("\nMRZ СТРОКИ")
        for line in result["строки"]:
            print(f"  {line}")

        print("\nРАСПОЗНАННЫЕ ПОЛЯ")
        for key, val in result["поля"].items():
            print(f"  {key:<20} {val}")

