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
        "--data_file", type=str, default="data.zip", help="Название скачанного датасета"
    )
    parser.add_argument(
        "--predict_data_file",
        type=str,
        default="predict_data.zip",
        help="Название скачанного файла с тестовыми данными",
    )

    args = parser.parse_args()

    download_dataset(
        file_id="1uj5hogXcbx2UzQ61u7PZbN4q5sBJcOk8",
        output_path=args.data_file,
    )

    download_dataset(
        file_id="19mBCLjsZa3xTzFHBMx-jlsjyfCxFsJ59",
        output_path=args.predict_data_file,
    )


if __name__ == "__main__":
    main()
