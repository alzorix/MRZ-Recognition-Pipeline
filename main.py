import  downloading
from YOLO_detecting import process_all
from ocr_extractor import recognize_mrz
import os
import logging
import csv

CSV_FILE = "result.csv"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if __name__ == '__main__':
    if not os.path.exists('downloaded_photos'):
        if os.path.exists('data.json'):
             logger.info('Downloading photos')
             downloading.setup_cache()
             downloading.parse_data()
        else:
             logger.error('Data not found')

    results = process_all()
    fieldnames = [
        'тип_документа',
        'страна_выдачи',
        'фамилия',
        'имя',
        'номер_документа',
        'гражданство',
        'дата_рождения',
        'пол',
        'срок_действия',
        'Контрольные_цифры'
    ]


    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in results:
            result = recognize_mrz(path)

            print("\nMRZ СТРОКИ")
            for line in result["строки"]:
                print(f"  {line}")

            print("\nРАСПОЗНАННЫЕ ПОЛЯ")
            writer.writerow(result["поля"])
            for key, val in result["поля"].items():
                print(f"  {key:<20} {val}")






