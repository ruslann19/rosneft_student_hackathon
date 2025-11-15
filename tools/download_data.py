# tools/download_data.py
import argparse

import gdown


def download_dataset(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"

    print("Скачивание данных...")
    gdown.download(url, output_path, quiet=False)


def main():
    parser = argparse.ArgumentParser(description="Скачивание данных с диска")
    parser.add_argument(
        "--zip_file", type=str, default="data.zip", help="Название скачанного файла"
    )

    args = parser.parse_args()

    download_dataset(
        file_id="1uj5hogXcbx2UzQ61u7PZbN4q5sBJcOk8",
        output_path=args.zip_file,
    )


if __name__ == "__main__":
    main()
