# tools/download_data.py
import os
import zipfile
from pathlib import Path

import gdown


def download_dataset(file_id, output_path, extract_to):
    url = f"https://drive.google.com/uc?id={file_id}"

    print("Скачивание данных...")
    gdown.download(url, output_path, quiet=False)

    print("Распаковка...")
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(output_path)
    print(f"Данные сохранены в: {Path(extract_to).resolve()}")


if __name__ == "__main__":
    download_dataset(
        file_id="1uj5hogXcbx2UzQ61u7PZbN4q5sBJcOk8",
        output_path="data.zip",
        extract_to="data",
    )
