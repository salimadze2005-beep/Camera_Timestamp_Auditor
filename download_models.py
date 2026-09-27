from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Загрузить публичные модели EasyOCR, необходимые для базового пайплайна."
    )
    parser.add_argument("--gpu", action="store_true", help="Инициализировать EasyOCR с CUDA")
    parser.add_argument("--force", action="store_true", help="Повторно проверить/скачать кэш моделей")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_dir = root / "models" / "easyocr"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        import easyocr
    except ImportError as exc:
        raise SystemExit("EasyOCR не установлен. Выполните: pip install -r requirements.txt") from exc

    print(f"Каталог моделей: {model_dir}")
    print("Проверка публичных моделей EasyOCR...")
    # Reader сам скачивает официальные detector/recognizer weights при их отсутствии.
    # Английский набор выбран потому, что таймштампы состоят из цифр и латинских разделителей.
    easyocr.Reader(
        ["en"],
        gpu=args.gpu,
        model_storage_directory=str(model_dir),
        download_enabled=True,
        verbose=True,
    )
    print("Модели EasyOCR готовы.")
    print("Кастомные .pt/.pth веса в проект не входят и для базового запуска не нужны.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
