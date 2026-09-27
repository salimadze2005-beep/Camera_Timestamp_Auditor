import os
import re
import csv
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytz
import torch
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
import gc

# ==========================================
# 1. ПАРСЕР ДАТЫ И ВРЕМЕНИ
# ==========================================

def ordered_dedup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def get_date_candidates(s):
    candidates = []
    if len(s) >= 4:
        candidates.append((s[:2], s[2:4]))
        candidates.append((s[-4:-2], s[-2:]))
    for i in range(len(s)):
        c = s[:i] + s[i + 1:]
        if len(c) >= 4:
            candidates.append((c[:2], c[2:4]))
            candidates.append((c[-4:-2], c[-2:]))
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            c = s[:i] + s[i + 1:j] + s[j + 1:]
            if len(c) >= 4:
                candidates.append((c[:2], c[2:4]))
                candidates.append((c[-4:-2], c[-2:]))
    if len(s) == 3:
        candidates.append((f"0{s[0]}", s[1:3]))
        candidates.append((s[0:2], f"0{s[2]}"))
    if len(s) == 2:
        candidates.append((f"0{s[0]}", f"0{s[1]}"))
    return ordered_dedup(candidates)

def get_time_candidates(s):
    candidates = []
    if len(s) >= 6:
        candidates.append((s[:2], s[2:4], s[4:6]))
        candidates.append((s[-6:-4], s[-4:-2], s[-2:]))
    for i in range(len(s)):
        c = s[:i] + s[i + 1:]
        if len(c) >= 6:
            candidates.append((c[:2], c[2:4], c[4:6]))
            candidates.append((c[-6:-4], c[-4:-2], c[-2:]))
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            c = s[:i] + s[i + 1:j] + s[j + 1:]
            if len(c) >= 6:
                candidates.append((c[:2], c[2:4], c[4:6]))
                candidates.append((c[-6:-4], c[-4:-2], c[-2:]))
    if len(s) == 5:
        candidates.append((s[0:2], s[2:4], f"0{s[4]}"))
        candidates.append((s[0:2], f"0{s[2]}", s[3:5]))
        candidates.append((f"0{s[0]}", s[1:3], s[3:5]))
    if len(s) == 4:
        candidates.append((s[0:2], s[2:4], "00"))
    return ordered_dedup(candidates)

def is_valid_date(year, month, day):
    try:
        y, m, d = int(year), int(month), int(day)
        if y < 2000 or y > 2030: return False
        if m < 1 or m > 12: return False
        if d < 1 or d > 31: return False
        return True
    except:
        return False

def is_valid_time(hour, minute, second="00"):
    try:
        h, m, s = int(hour), int(minute), int(second)
        if h < 0 or h > 23: return False
        if m < 0 or m > 59: return False
        if s < 0 or s > 59: return False
        return True
    except:
        return False

def is_valid_datetime(year, month, day, hour, minute, second):
    return is_valid_date(year, month, day) and is_valid_time(hour, minute, second)

def format_datetime_from_groups(groups, original_text):
    if len(groups) == 6:
        g1, g2, g3, g4, g5, g6 = groups
        if len(g1) == 4:
            year, month, day = g1, g2, g3
        else:
            day, month, year = g1, g2, g3
        hour, minute, second = g4, g5, g6
    elif len(groups) == 4:
        g1, g2, g3, g4 = groups
        if len(g1) == 4:
            year, month, day = g1, g2, g3
        else:
            day, month, year = g1, g2, g3
        if ':' in g4 or len(g4) >= 4:
            time_match = re.search(r'(\d{1,2}):?(\d{2})', g4)
            if time_match:
                hour, minute = time_match.groups()
                second = "00"
            else:
                return None
        else:
            return None
    else:
        return None

    if not is_valid_datetime(year, month, day, hour, minute, second): return None
    if 'PM' in original_text.upper():
        h = int(hour)
        if h < 12: hour = str(h + 12)
    elif 'AM' in original_text.upper():
        h = int(hour)
        if h == 12: hour = "00"
    return f"{year}-{month.zfill(2)}-{day.zfill(2)} {str(hour).zfill(2)}:{minute.zfill(2)}:{second.zfill(2)}"

def extract_datetime_robust(raw_text):
    text = raw_text.upper()
    is_pm, is_am = 'PM' in text, 'AM' in text
    text_clean = text.replace('AM', '').replace('PM', '')

    replacements = {"O": "0", "I": "1", "L": "1", "|": "1", "S": "5", "B": "8", "Z": "2", "T": "7", "G": "6", "A": "4", "Q": "0"}
    for old, new in replacements.items():
        text_clean = text_clean.replace(old, new)

    cleaned = re.sub(r'[-/.,\\]', '-', re.sub(r'\s+', '', text_clean))
    patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})-?(\d{1,2}):?(\d{2}):?(\d{2})',
        r'(\d{1,2})-(\d{1,2})-(\d{4})-?(\d{1,2}):?(\d{2}):?(\d{2})'
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            dt = format_datetime_from_groups(match.groups(), raw_text)
            if dt: return dt

    digits = re.sub(r'\D', '', text_clean)
    match = re.search(r'(20[23]\d)', digits)
    if not match: match = re.search(r'(0[23]\d)', digits)

    if match:
        raw_year = match.group(1)
        year = "20" + raw_year[1:] if len(raw_year) == 3 else raw_year
        idx, year_len = match.start(), len(raw_year)

        def check_and_format(d, m, h_str, m_str, s_str):
            if not is_valid_date(year, m, d): return None
            try:
                h = int(h_str)
                if is_pm and h < 12: h += 12
                if is_am and h == 12: h = 0
                h_24 = str(h).zfill(2)
            except ValueError:
                return None
            if is_valid_datetime(year, m, d, h_24, m_str, s_str):
                return f"{year}-{m.zfill(2)}-{d.zfill(2)} {h_24}:{m_str.zfill(2)}:{s_str.zfill(2)}"
            return None

        if idx >= 3:
            pre_year, post_year = digits[:idx], digits[idx + year_len:]
            for d1, d2 in get_date_candidates(pre_year):
                for day, month in [(d1, d2), (d2, d1)]:
                    if not post_year: continue
                    for h, mins, sec in get_time_candidates(post_year):
                        res = check_and_format(day, month, h, mins, sec)
                        if res: return res
        else:
            post_year = digits[idx + year_len:]
            for split_idx in [4, 5, 6, 7, 3, 8]:
                if split_idx > len(post_year): continue
                date_part, time_part = post_year[:split_idx], post_year[split_idx:]
                if not time_part: continue
                for d1, d2 in get_date_candidates(date_part):
                    month, day = d1, d2
                    for h, mins, sec in get_time_candidates(time_part):
                        res = check_and_format(day, month, h, mins, sec)
                        if res: return res
    return None

# ==========================================
# 2. СВЯЗЬ ВРЕМЕНИ С ЭТАЛОНОМ
# ==========================================

def try_smart_adjustment(parsed_dt_str, ref_time_str, tz_str, tolerance):
    try:
        if not parsed_dt_str or parsed_dt_str.strip() == "":
            return "", None, "PARSE_FAILED"

        try:
            parsed_dt_naive = pd.to_datetime(parsed_dt_str)
        except (ValueError, TypeError):
            return parsed_dt_str, None, "PARSE_FAILED"

        tz_camera = pytz.timezone(tz_str)
        parsed_dt_local = tz_camera.localize(parsed_dt_naive)
        parsed_dt_utc = parsed_dt_local.astimezone(pytz.utc)

        ref_dt_utc = pd.to_datetime(ref_time_str)
        if ref_dt_utc.tzinfo is None:
            ref_dt_utc = pytz.utc.localize(ref_dt_utc)

        delta_minutes = abs((parsed_dt_utc - ref_dt_utc).total_seconds()) / 60.0
        delta_minutes = round(delta_minutes, 2)

        if delta_minutes <= tolerance:
            return parsed_dt_str, delta_minutes, "OK"
        if delta_minutes > 1300:
            return parsed_dt_str, delta_minutes, "NEED_REVIEW"

        mod_60 = delta_minutes % 60
        if mod_60 <= tolerance or mod_60 >= (60 - tolerance):
            return parsed_dt_str, delta_minutes, "TIMEZONE_MISMATCH"

        return parsed_dt_str, delta_minutes, "TIME_WRONG"

    except Exception as e:
        return parsed_dt_str, None, f"TECH_ERROR: {str(e)}"

# ==========================================
# 3. ГЛАВНЫЙ ИНФЕРЕНС-ПАЙПЛАЙН С CUSTOM EASYOCR
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Auditing pipeline using Custom Fine-Tuned EasyOCR Model")
    parser.add_argument("--manifest", required=True, help="Путь к манифесту")
    parser.add_argument("--images-dir", required=True, help="Папка с картинками selected")
    parser.add_argument("--out", required=True, help="Выходной CSV с результатами")
    parser.add_argument("--yolo-model", required=True, help="Путь к весам YOLO (.pt)")
    parser.add_argument("--ocr-model", required=True, help="Путь к весам Custom EasyOCR (.pth)")
    parser.add_argument("--tolerance", type=float, default=10.0, help="Допуск отклонения")
    parser.add_argument("--timezone-default", default="Asia/Yekaterinburg", help="Таймзона по умолчанию")
    parser.add_argument("--failed-crops-dir", default="failed_crops", help="Папка для сохранения кропов, где OCR ошибся")
    parser.add_argument("--save-every", type=int, default=25, help="Save partial CSV every N processed rows")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Загрузка EasyOCR компонентов
    try:
        from easyocr.utils import CTCLabelConverter
        from easyocr.model.vgg_model import Model
    except ImportError:
        raise ImportError("easyocr library is not installed. Please run 'pip install easyocr'")

    print(f"Загрузка модели Custom EasyOCR: {args.ocr_model}")
    checkpoint = torch.load(args.ocr_model, map_location=device)
    
    characters = checkpoint['characters']
    num_class = checkpoint['num_class']
    converter = CTCLabelConverter(characters)
    
    ocr_model = Model(input_channel=1, output_channel=256, hidden_size=256, num_class=num_class).to(device)
    ocr_model.load_state_dict(checkpoint['model_state_dict'])
    ocr_model.eval()

    print(f"Загрузка YOLO: {args.yolo_model}")
    detector = YOLO(args.yolo_model)

    results = []
    columns_order = [
        "image_path", "camera_id", "split", "reference_time", "timezone",
        "detected_bbox", "detector_confidence", "ocr_text_ready_model",
        "ocr_text_trained_model", "parsed_time", "delta_minutes",
        "status", "status_reason", "debug_image_path", "crop_path", "comment"
    ]

    def save_partial():
        if not results:
            return
        pd.DataFrame(results)[columns_order].to_csv(args.out, index=False, encoding='utf-8-sig')

    def add_result(row):
        results.append(row)
        if args.save_every > 0 and len(results) % args.save_every == 0:
            save_partial()

    try:
        from tqdm import tqdm
    except ImportError:
        class tqdm_fallback:
            def __init__(self, iterable, desc=''):
                self.iterable = iterable
                self.desc = desc
                self.total = len(iterable) if hasattr(iterable, '__len__') else None
            def __iter__(self):
                print(f"Starting: {self.desc}")
                for i, item in enumerate(self.iterable):
                    if self.total and (i == 0 or (i + 1) % max(1, self.total // 10) == 0 or i == self.total - 1):
                        print(f"{self.desc} Progress: {i+1}/{self.total}")
                    yield item
            def set_postfix(self, *args, **kwargs):
                pass
            def set_description(self, *args, **kwargs):
                pass
        def tqdm(iterable, *args, **kwargs):
            desc = kwargs.get('desc', '')
            return tqdm_fallback(iterable, desc=desc)

    print("Загрузка манифеста...")
    manifest_data = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            manifest_data.append(row)

    print(f"Найдено {len(manifest_data)} изображений. Запуск аудита...")

    for row in tqdm(manifest_data, desc="Auditing images"):
        filename = Path(row.get("selected_image_path", "")).name
        img_path = str(Path(args.images_dir) / filename)
        camera_id = row.get("camera_id", "")
        split = row.get("split", "test")
        ref_time = row.get("reference_time", "")
        tz_str = row.get("timezone", args.timezone_default)

        out_row = {
            "image_path": img_path,
            "camera_id": camera_id,
            "split": split,
            "reference_time": ref_time,
            "timezone": tz_str,
            "detected_bbox": "",
            "detector_confidence": "",
            "ocr_text_ready_model": "",
            "ocr_text_trained_model": "",
            "parsed_time": "",
            "delta_minutes": "",
            "status": "",
            "status_reason": "",
            "debug_image_path": "",
            "crop_path": "",
            "comment": ""
        }

        if not ref_time:
            out_row["status"] = "REFERENCE_TIME_MISSING"
            out_row["status_reason"] = "Нет reference_time в манифесте"
            add_result(out_row)
            continue

        if not os.path.exists(img_path):
            found = False
            for s in ["train", "val", "test", split]:
                test_path = str(Path(args.images_dir) / s / "images" / filename)
                if os.path.exists(test_path):
                    img_path = test_path
                    found = True
                    break
            if not found:
                out_row["status"] = "TECH_ERROR"
                out_row["status_reason"] = f"Файл не найден: {img_path}"
                add_result(out_row)
                continue
            
        img_array = np.fromfile(img_path, dtype=np.uint8)
        try:
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except cv2.error as e:
            out_row["status"] = "TECH_ERROR"
            out_row["status_reason"] = f"OpenCV decode error: {str(e)}"
            add_result(out_row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        if image is None:
            out_row["status"] = "TECH_ERROR"
            out_row["status_reason"] = "Не удалось декодировать картинку"
            add_result(out_row)
            continue

        # Детекция YOLO
        pred = detector.predict(image, verbose=False)[0]
        if len(pred.boxes) == 0:
            out_row["status"] = "TIMESTAMP_NOT_FOUND"
            out_row["status_reason"] = "Детектор не нашел bbox"
            add_result(out_row)
            continue

        best_box = max(pred.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        conf = float(best_box.conf[0])

        out_row["detected_bbox"] = f"{x1},{y1},{x2},{y2}"
        out_row["detector_confidence"] = round(conf, 4)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            out_row["status"] = "TECH_ERROR"
            out_row["status_reason"] = "Вырезанный кроп пустой"
            add_result(out_row)
            continue

        # Препроцессинг кропа под EasyOCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 64), interpolation=cv2.INTER_CUBIC)
        gray = gray.astype(np.float32) / 255.0
        
        img_tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0).to(device)

        # Распознавание нашей моделью EasyOCR
        with torch.no_grad():
            preds = ocr_model(img_tensor, None)  # [1, 63, num_class]
            _, max_indices = preds.max(2)
            
            flat_preds = max_indices.flatten().cpu().numpy()
            pred_lengths = np.array([max_indices.size(1)], dtype=np.int32)
            
            raw_text = converter.decode_greedy(flat_preds, pred_lengths)[0]
        
        out_row["ocr_text_trained_model"] = raw_text

        if not raw_text.strip():
            out_row["status"] = "OCR_FAILED"
            out_row["status_reason"] = "OCR вернул пустое распознавание"
            
            # Сохраняем кроп ошибки
            failed_dir = Path(args.failed_crops_dir)
            failed_dir.mkdir(exist_ok=True, parents=True)
            crop_name = f"{Path(filename).stem}_ref_{ref_time.replace(':', '-').replace(' ', '_')}_ocr_empty.jpg"
            crop_path = failed_dir / crop_name
            cv2.imencode('.jpg', crop)[1].tofile(str(crop_path))
            out_row["crop_path"] = str(crop_path)
            
            add_result(out_row)
            continue

        parsed_dt_str = extract_datetime_robust(raw_text)
        if not parsed_dt_str:
            out_row["status"] = "PARSE_FAILED"
            out_row["status_reason"] = f"Парсер не распознал дату из текста: '{raw_text}'"
            
            # Сохраняем кроп ошибки
            failed_dir = Path(args.failed_crops_dir)
            failed_dir.mkdir(exist_ok=True, parents=True)
            clean_ocr = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_text)
            crop_name = f"{Path(filename).stem}_ref_{ref_time.replace(':', '-').replace(' ', '_')}_ocr_{clean_ocr}.jpg"
            crop_path = failed_dir / crop_name
            cv2.imencode('.jpg', crop)[1].tofile(str(crop_path))
            out_row["crop_path"] = str(crop_path)
            
            add_result(out_row)
            continue

        out_row["parsed_time"] = parsed_dt_str

        fixed_parsed_time, delta_minutes, status = try_smart_adjustment(
            parsed_dt_str, ref_time, tz_str, args.tolerance
        )

        out_row["parsed_time"] = fixed_parsed_time
        out_row["delta_minutes"] = delta_minutes
        out_row["status"] = status

        if status == "OK":
            out_row["status_reason"] = f"Дельта {delta_minutes} мин. (допуск {args.tolerance})"
        else:
            if status == "TIMEZONE_MISMATCH":
                out_row["status_reason"] = f"Сдвиг на целое число часов (дельта {delta_minutes} мин). Неверная таймзона"
            elif status == "TIME_WRONG":
                out_row["status_reason"] = f"Дельта {delta_minutes} мин. превышает допуск"
            elif status == "NEED_REVIEW":
                out_row["status_reason"] = f"Дельта {delta_minutes} мин. (>1 дня). Вероятно OCR ошибся в дате"
            else:
                out_row["status_reason"] = status

            # Сохраняем кроп ошибки (так как статус не OK)
            failed_dir = Path(args.failed_crops_dir)
            failed_dir.mkdir(exist_ok=True, parents=True)
            clean_ref = re.sub(r'[^a-zA-Z0-9_-]', '_', ref_time)
            clean_ocr = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_text)
            crop_name = f"{Path(filename).stem}_ref_{clean_ref}_ocr_{clean_ocr}.jpg"
            crop_path = failed_dir / crop_name
            cv2.imencode('.jpg', crop)[1].tofile(str(crop_path))
            out_row["crop_path"] = str(crop_path)

        add_result(out_row)
        del image, crop, img_tensor
        if 'pred' in locals():
            del pred
        if len(results) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Сохранение результатов в {args.out}...")
    df_results = pd.DataFrame(results)
    
    df_results = df_results[columns_order]
    df_results.to_csv(args.out, index=False, encoding='utf-8-sig')

    print("\n=== ИТОГОВАЯ СТАТИСТИКА CUSTOM EASYOCR ===")
    print(df_results['status'].value_counts())
    print("\nИнференс завершен успешно!")

if __name__ == "__main__":
    main()
