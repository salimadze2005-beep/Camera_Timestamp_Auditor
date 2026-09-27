# Camera Timestamp Auditor

**OCR / Computer Vision система для автоматической проверки даты и времени на кадрах камер видеонаблюдения.**

Камеры часто выводят OSD timestamp прямо поверх изображения. Если часы камеры сбились, запись становится сложнее сопоставить с другими событиями. Система распознаёт timestamp, исправляет типичные OCR-ошибки, учитывает часовой пояс и сравнивает полученное время с эталонным.

> Основной фокус проекта: OCR pipeline, post-processing распознанного текста, генерация кандидатов timestamp и timezone-aware validation.

## Что делает проект

Pipeline автоматизирует проверку времени на изображениях:

1. Получает кадр, эталонное время, timezone и предполагаемое положение OSD.
2. Выполняет OCR timestamp на нескольких областях изображения.
3. Нормализует распознанную строку и исправляет типичные OCR-подмены.
4. Разбирает разные форматы даты и времени и формирует несколько допустимых кандидатов.
5. Сравнивает каждый кандидат с эталонным временем с учётом часового пояса.
6. Повторно обрабатывает сложные кадры и объединяет результаты нескольких проходов.
7. Сохраняет итоговый статус и диагностическую информацию в CSV.

Основные статусы: `OK`, `TIMEZONE_MISMATCH`, `TIME_WRONG`, `NEED_REVIEW` и технические статусы ошибок.

## Моя роль - Computer Vision Intern / ML Engineer

Проект выполнялся на производственной практике в **АО «Уфанет»**, команда из двух человек. Я работал над OCR/CV pipeline и логикой проверки timestamp:

- подготавливал и размечал данные для обучения и проверки OCR pipeline;
- участвовал в обучении и валидации специализированной OCR recognition-модели;
- разработал **parser/post-processing** для нескольких форматов даты и времени;
- добавил исправление типичных OCR-ошибок, включая **O/0, I/l/1, S/5, B/8**;
- реализовал восстановление разделителей и генерацию нескольких кандидатов timestamp вместо жёсткого разбора одной строки;
- реализовал **timezone-aware validation** распознанного времени относительно эталонного значения;
- анализировал проблемные OCR-примеры и добавлял повторную обработку сложных кадров;
- итоговый pipeline достиг **95,5% по внутренней метрике** на датасете из **2000 изображений**; метрика учитывала корректное распознавание timestamp и выявление timezone mismatch.


## Архитектура

```text
Image + reference time + timezone
              │
              ▼
        OCR / EasyOCR
              │
              ▼
      Text normalization
              │
              ▼
 OCR error correction
              │
              ▼
 Timestamp candidates
              │
              ▼
Timezone-aware validation
              │
       ┌──────┴──────┐
       ▼             ▼
      OK       retry / review
                     │
                     ▼
             Ensemble result
                     │
                     ▼
                    CSV
```

## Post-processing

В проекте реализованы:

- замены OCR-символов `O → 0`, `I/L/| → 1`, `S → 5`, `B → 8` и др.;
- обработка `AM/PM`;
- поддержка разных вариантов `YMD / DMY / MDY`;
- генерация кандидатов из частично повреждённых строк;
- использование reference time для восстановления неполных timestamp;
- проверка weekday;
- timezone-aware сравнение с эталоном;
- классификация расхождений по типу ошибки.

## Технологии

**Computer Vision / OCR**
- Python
- PyTorch
- OpenCV
- EasyOCR
- OCR recognition / CTC
- image preprocessing

**Data / validation**
- NumPy
- Pandas
- pytz
- synthetic OCR data
- error analysis
- CSV pipelines
- unit tests

## Структура репозитория

```text
.
├── src/
│   ├── run_easyocr.py
│   ├── run_custom_easyocr.py
│   ├── postprocess_timestamp_results.py
│   ├── rerun_failed_easyocr.py
│   ├── make_ocr_ensemble.py
│   ├── train_easyocr.py
│   ├── generate_synthetic_ocr_data.py
│   └── export_problem_gallery.py
│
├── config/
│   ├── manifest_template.csv
│   └── requirements_easyocr.txt
│
├── tests/
│   ├── test_timestamp_logic.py
│   └── test_postprocess_csv.py
│
├── run_pipeline.py
├── RUN_PIPELINE.ps1
└── RUN_PIPELINE.sh
```

## Запуск проекта

Нужен Python 3.10–3.12. GPU не обязателен.

Windows:

```powershell
.\RUN_PIPELINE.ps1 -ImagesDir "D:\camera_frames" -Manifest "D:\manifest.csv"
```

Linux:

```bash
./RUN_PIPELINE.sh --images-dir /data/camera_frames --manifest /data/manifest.csv
```

Пример manifest:

```csv
selected_image_path,reference_time,timezone,camera_id,timestamp_candidate_position
frame001.jpg,2026-06-08T14:31:22+05:00,Asia/Yekaterinburg,cam001,top-right
```

Тесты логики без моделей и реальных кадров:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

