import json
import requests
import requests_cache
import os
import logging
from tqdm import tqdm
from pathlib import Path

main_logger = logging.getLogger('download_module')
main_logger.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

CACHE_DIR = "requests_cache"
DOWNLOAD_DIR = "downloaded_photos"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def setup_cache():
    """Настройка кэша для requests"""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    requests_cache.install_cache(
        cache_name=os.path.join(CACHE_DIR, 'photos_cache'),
        backend='sqlite'
    )

def download_photo(filename: str, url: str, download_dir: str) -> bool:
    """Скачивание одного фото"""
    file_path = os.path.join(download_dir, filename)

    if os.path.exists(file_path):
        main_logger.debug(f"Already exists, skipping: {file_path}")
        return True

    try:
        response = requests.get(url, stream=True, headers=HEADERS, timeout=15)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):  # 8KB чанки
                f.write(chunk)

        main_logger.debug(f"Downloaded: {file_path}")
        return True

    except requests.exceptions.HTTPError as e:
        main_logger.error(f"HTTP error {e.response.status_code} for {filename}: {e}")
        return False
    except requests.exceptions.Timeout:
        main_logger.error(f"Timeout while downloading: {filename}")
        return False
    except Exception as e:
        main_logger.error(f"Failed to download {filename}: {e}")
        return False


def parse_data(filename_data: str = "data.json", download_dir: str = DOWNLOAD_DIR):
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    with open(filename_data, "r", encoding="utf-8") as f:  # используем параметр
        data = json.load(f)

    images = data['images']
    success_count = 0
    failed = []

    for document in tqdm(images, desc="Downloading photos", unit="img"):
        filename = document['filename']
        url = document['url']

        if download_photo(filename, url, download_dir):
            success_count += 1
        else:
            failed.append(filename)

    main_logger.info(f"Done: {success_count}/{len(images)} downloaded successfully.")

    if failed:
        main_logger.warning(f"Failed ({len(failed)}): {failed}")


if __name__ == '__main__':
    setup_cache()
    parse_data()