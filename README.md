# Camera Timestamp Auditor

Инструмент для проверки времени, нанесённого поверх кадров с камер. Проект читает список изображений из CSV, распознаёт таймштамп, приводит его к нормализованному времени, сравнивает с эталонным временем и сохраняет результат аудита в CSV.

В публичной версии нет исходных изображений, production-манифестов, результатов предыдущих запусков и обученных `.pt`/`.pth` файлов. Для рабочего запуска используются публичные модели EasyOCR, которые скачиваются при первой настройке.

## Возможности

- обработка изображений по CSV-манифесту;
- поиск таймштампа по ожидаемой позиции в кадре;
- опциональная поддержка собственного YOLO-детектора;
- несколько вариантов предобработки изображения;
- повторное распознавание проблемных строк;
- разбор повреждённых OCR-строк с датой и временем;
- сравнение с эталонным временем с учётом часового пояса;
- итоговый CSV с `OK`, `TIMEZONE_MISMATCH`, `TIME_WRONG`, `NEED_REVIEW` и техническими статусами;
- скрипты обучения собственного распознавателя сохранены в `src/`, но его веса в репозиторий не входят.

## Требования

- Python 3.10–3.12;
- Windows или Linux;
- CUDA не обязательна.

## Быстрый запуск на Windows

```powershell
.\RUN_PIPELINE.ps1 -ImagesDir "D:\camera_frames" -Manifest "D:\manifest.csv"
```

С GPU:

```powershell
.\RUN_PIPELINE.ps1 -ImagesDir "D:\camera_frames" -Manifest "D:\manifest.csv" -Gpu
```

Скрипт создаёт `.venv`, устанавливает зависимости и вызывает `download_models.py`. EasyOCR скачивает официальные веса в `models/easyocr`.

## Ручная установка

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python download_models.py
python run_pipeline.py --images-dir D:\camera_frames --manifest D:\manifest.csv
```

Для Linux команды те же, кроме активации окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
python run_pipeline.py --images-dir /data/camera_frames --manifest /data/manifest.csv
```

## Формат манифеста

Шаблон находится в `config/manifest_template.csv`.

```csv
selected_image_path,reference_time,timezone,camera_id,timestamp_candidate_position
frame001.jpg,2026-06-08T14:31:22+05:00,Asia/Yekaterinburg,cam001,top-right
```

`timestamp_candidate_position` принимает значения вроде `top-right`, `top-left`, `bottom-right` или `bottom-left`.

## Кастомный детектор

Базовый режим не требует YOLO-весов. Если у вас есть собственный детектор области таймштампа, его можно подключить:

```bash
python run_pipeline.py --images-dir ./frames --manifest ./manifest.csv --yolo-model ./models/timestamp_detector.pt
```

Путь к весам не коммитится: `.gitignore` исключает `.pt`, `.pth`, `.onnx` и `.safetensors`.

## Структура

```text
config/                 шаблон манифеста и список зависимостей старого training-пайплайна
data/images/            локальные изображения, не попадают в Git
models/                 скачанные модели, не попадают в Git
results/                результаты запусков, не попадают в Git
src/run_easyocr.py      основной публичный inference
src/rerun_failed_easyocr.py
src/postprocess_timestamp_results.py
src/make_ocr_ensemble.py
src/train_easyocr.py    обучение собственного OCR
src/run_custom_easyocr.py  запуск собственного OCR checkpoint
run_pipeline.py         оркестратор
RUN_PIPELINE.ps1        установка и запуск на Windows
RUN_PIPELINE.sh         установка и запуск на Linux
download_models.py      загрузка официальных моделей EasyOCR
```

## Проверка кода

Тесты логики не требуют весов:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q .
```

Полный OCR-тест требует интернет при первом запуске, потому что EasyOCR должен скачать модель. В проект намеренно не включены тестовые изображения или чужие данные.

## Приватность

Не добавляйте в репозиторий реальные кадры камер, внутренние CSV, failed crops или обученные веса. Для рабочих данных используйте каталоги, исключённые через `.gitignore`.
