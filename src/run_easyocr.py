from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from postprocess_timestamp_results import choose_candidate

ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-/:. "


def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_image(images_dir: Path, image_path: str, split: str = "") -> Path | None:
    filename = Path(str(image_path)).name
    candidates = [images_dir / filename]
    if split:
        candidates.append(images_dir / split / "images" / filename)
    for sub in ("train", "val", "test"):
        candidates.append(images_dir / sub / "images" / filename)
    return next((p for p in candidates if p.is_file()), None)


def position_boxes(image: np.ndarray, position: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    h, w = image.shape[:2]
    pos = str(position or "").lower().replace("_", "-")
    top = "bottom" not in pos
    right = "right" in pos

    # Несколько размеров дают устойчивость к разным OSD и разрешениям камер.
    specs = [(0.58, 0.18), (0.48, 0.13), (0.72, 0.24)]
    out = []
    for idx, (wf, hf) in enumerate(specs, 1):
        cw, ch = max(220, int(w * wf)), max(70, int(h * hf))
        x1 = max(0, w - cw) if right else 0
        x2 = w if right else min(w, cw)
        y1 = 0 if top else max(0, h - ch)
        y2 = min(h, ch) if top else h
        out.append((f"position_{idx}", (x1, y1, x2, y2)))
    # Полные верхняя/нижняя полосы — fallback, если положение указано неточно.
    sh = max(90, int(h * 0.20))
    out.append(("full_strip", (0, 0 if top else h - sh, w, sh if top else h)))
    return out


def yolo_box(image: np.ndarray, model_path: str | None):
    if not model_path:
        return None
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {path}")
    from ultralytics import YOLO

    model = YOLO(str(path))
    pred = model.predict(image, verbose=False)[0]
    if len(pred.boxes) == 0:
        return None
    best = max(pred.boxes, key=lambda b: float(b.conf[0]))
    box = tuple(map(int, best.xyxy[0]))
    return box, float(best.conf[0])


def preprocess(crop: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    pad = cv2.copyMakeBorder(crop, 10, 10, 16, 16, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(pad, cv2.COLOR_BGR2GRAY)
    yield "raw", pad
    yield "gray", gray
    yield "gray3", cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield "clahe3", cv2.resize(clahe, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu3", cv2.resize(otsu, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    yield "otsu_inv3", cv2.resize(cv2.bitwise_not(otsu), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def read_best(reader, image: np.ndarray, boxes, ref_time: str, timezone: str, tolerance: float):
    best = None
    for box_name, (x1, y1, x2, y2) in boxes:
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        for variant_name, variant in preprocess(crop):
            texts = reader.readtext(variant, detail=0, paragraph=False, allowlist=ALLOWLIST)
            raw = " ".join(texts).strip()
            if not raw:
                continue
            try:
                dt, delta, status, reason = choose_candidate(
                    raw, ref_time, timezone, tolerance, allow_reference_repair=False
                )
            except Exception:
                continue
            if dt is None:
                rank = 5
                delta_value = 10**12
            else:
                rank = {"OK": 0, "TIMEZONE_MISMATCH": 1, "TIME_WRONG": 2, "NEED_REVIEW": 3}.get(status, 4)
                delta_value = float(delta) if delta is not None else 10**12
            candidate = (rank, delta_value, raw, dt, status, reason, box_name, variant_name, (x1, y1, x2, y2))
            if best is None or candidate[:2] < best[:2]:
                best = candidate
                if rank == 0:
                    return best
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Распознавание и аудит таймштампов камер")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--timezone-default", default="UTC")
    parser.add_argument("--yolo-model", default="", help="Необязательные кастомные YOLO-веса")
    parser.add_argument("--model-dir", default="models/easyocr")
    args = parser.parse_args()

    import easyocr

    root = Path(__file__).resolve().parent.parent
    model_dir = (root / args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        ["en"],
        gpu=args.gpu,
        model_storage_directory=str(model_dir),
        download_enabled=True,
        verbose=False,
    )

    with open(args.manifest, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    output = []
    images_dir = Path(args.images_dir)
    for row in tqdm(rows, desc="Аудит таймштампов"):
        image_path = find_image(images_dir, row.get("selected_image_path", ""), row.get("split", ""))
        ref_time = row.get("reference_time", "")
        timezone = row.get("timezone", "") or args.timezone_default
        result = {
            "image_path": str(image_path or row.get("selected_image_path", "")),
            "camera_id": row.get("camera_id", ""),
            "split": row.get("split", ""),
            "reference_time": ref_time,
            "timezone": timezone,
            "timestamp_candidate_position": row.get("timestamp_candidate_position", "top-right"),
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
            "comment": "",
        }
        if not ref_time:
            result.update(status="REFERENCE_TIME_MISSING", status_reason="В манифесте нет reference_time")
            output.append(result); continue
        if image_path is None:
            result.update(status="TECH_ERROR", status_reason="Файл изображения не найден")
            output.append(result); continue
        image = imread(image_path)
        if image is None:
            result.update(status="TECH_ERROR", status_reason="Не удалось декодировать изображение")
            output.append(result); continue

        boxes = []
        if args.yolo_model:
            det = yolo_box(image, args.yolo_model)
            if det:
                box, conf = det
                boxes.append(("yolo", box))
                result["detector_confidence"] = round(conf, 4)
        boxes.extend(position_boxes(image, row.get("timestamp_candidate_position", "top-right")))
        best = read_best(reader, image, boxes, ref_time, timezone, args.tolerance)
        if best is None:
            result.update(status="OCR_FAILED", status_reason="EasyOCR не вернул пригодный текст")
        else:
            _, _, raw, dt, status, reason, box_name, variant_name, box = best
            result["ocr_text_ready_model"] = raw
            result["detected_bbox"] = ",".join(map(str, box))
            result["comment"] = f"bbox={box_name}; preprocess={variant_name}"
            if dt is None:
                result.update(status="PARSE_FAILED", status_reason=f"Не удалось разобрать: {raw}")
            else:
                result["parsed_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    _, delta, status2, reason2 = choose_candidate(raw, ref_time, timezone, args.tolerance, allow_reference_repair=False)
                    result["delta_minutes"] = delta
                    result["status"] = status2
                    result["status_reason"] = reason2
                except Exception as exc:
                    result.update(status=status or "TECH_ERROR", status_reason=str(exc))
        output.append(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {out}")
    if output:
        print(pd.Series([r["status"] for r in output]).value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
