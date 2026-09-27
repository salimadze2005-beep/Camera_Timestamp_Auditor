import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


GOOD_STATUSES = {"OK", "TIMEZONE_MISMATCH"}


def safe_name(text, max_len=90):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "empty"


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", image)[1].tofile(str(path))


def find_image(images_dir, image_path, split):
    filename = Path(str(image_path)).name
    candidates = [
        Path(images_dir) / filename,
        Path(images_dir) / str(split) / "images" / filename,
    ]
    for sub in ["train", "val", "test"]:
        candidates.append(Path(images_dir) / sub / "images" / filename)
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_bbox(value):
    if pd.isna(value) or not str(value).strip():
        return None
    try:
        return [int(float(part)) for part in str(value).split(",")]
    except Exception:
        return None


def classify_problem(row):
    status = str(row.get("status", "UNKNOWN"))
    visibility = str(row.get("timestamp_visibility_category", ""))
    bbox = row.get("detected_bbox", "")
    ready = row.get("ocr_text_ready_model", "")
    trained = row.get("ocr_text_trained_model", "")
    ocr_text = "" if pd.isna(ready) else str(ready)
    if not ocr_text and not pd.isna(trained):
        ocr_text = str(trained)

    if status == "TIMESTAMP_NOT_FOUND":
        if "missing" in visibility:
            return "01_timestamp_probably_missing"
        return "02_detector_not_found_visible"
    if status == "PARSE_FAILED":
        if pd.isna(bbox) or not str(bbox).strip():
            return "03_parse_failed_no_bbox"
        if not ocr_text.strip():
            return "04_ocr_empty"
        return "05_parse_failed_garbage_text"
    if status == "TIME_WRONG":
        if re.search(r"[A-Za-zА-Яа-я]{2,}", ocr_text) or re.search(r"[:./-].*[:./-].*[:./-].*", ocr_text):
            return "06_time_wrong_ocr_suspect"
        return "07_time_wrong_maybe_real"
    if status == "NEED_REVIEW":
        if ocr_text.strip():
            return "08_need_review_bad_date_ocr"
        return "09_need_review_trained_only_or_empty"
    return f"99_{safe_name(status, 40)}"


def main():
    parser = argparse.ArgumentParser(description="Export problem images/crops for manual review")
    parser.add_argument("--results-csv", default="ensemble_hard_probe.csv")
    parser.add_argument("--manifest", default="selected_manifest.csv")
    parser.add_argument("--images-dir", default="selected")
    parser.add_argument("--out-dir", default="problem_gallery")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    manifest = pd.read_csv(args.manifest)
    add_cols = [
        "camera_id",
        "timestamp_visibility_category",
        "timestamp_candidate_position",
        "timestamp_color",
        "estimated_time_of_day",
        "quality_tags",
        "has_timestamp_candidate",
    ]
    merged = df.merge(manifest[[c for c in add_cols if c in manifest.columns]], on="camera_id", how="left")
    bad = merged[~merged["status"].isin(GOOD_STATUSES)].copy()

    out_dir = Path(args.out_dir)
    originals_dir = out_dir / "originals"
    crops_dir = out_dir / "crops"
    annotated_dir = out_dir / "annotated"
    for d in [originals_dir, crops_dir, annotated_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in bad.iterrows():
        image_path = find_image(args.images_dir, row.get("image_path", ""), row.get("split", ""))
        problem = classify_problem(row)
        camera_id = safe_name(row.get("camera_id", "camera"), 35)
        status = safe_name(row.get("status", "status"), 30)
        position = safe_name(row.get("timestamp_candidate_position", "pos"), 20)
        delta = safe_name(row.get("delta_minutes", "na"), 20)
        ocr = row.get("ocr_text_ready_model", "")
        if pd.isna(ocr) or not str(ocr).strip():
            ocr = row.get("ocr_text_trained_model", "")
        ocr_short = safe_name(ocr, 45)
        base_name = f"{problem}__{status}__{position}__delta_{delta}__{camera_id}__ocr_{ocr_short}.jpg"

        out_original = originals_dir / problem / base_name
        out_crop = crops_dir / problem / base_name
        out_annotated = annotated_dir / problem / base_name

        image_found = image_path is not None
        crop_written = False
        annotated_written = False

        if image_path is not None:
            out_original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, out_original)
            image = imread(image_path)
            bbox = parse_bbox(row.get("detected_bbox", ""))
            if image is not None and bbox:
                x1, y1, x2, y2 = bbox
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = image[y1:y2, x1:x2]
                if crop.size:
                    imwrite(out_crop, crop)
                    crop_written = True
                annotated = image.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated,
                    str(row.get("status", "")),
                    (max(0, x1), max(30, y1 + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                imwrite(out_annotated, annotated)
                annotated_written = True

        rows.append({
            "problem": problem,
            "camera_id": row.get("camera_id", ""),
            "split": row.get("split", ""),
            "status": row.get("status", ""),
            "reference_time": row.get("reference_time", ""),
            "delta_minutes": row.get("delta_minutes", ""),
            "detected_bbox": row.get("detected_bbox", ""),
            "detector_confidence": row.get("detector_confidence", ""),
            "ocr_text_ready_model": row.get("ocr_text_ready_model", ""),
            "ocr_text_trained_model": row.get("ocr_text_trained_model", ""),
            "parsed_time": row.get("parsed_time", ""),
            "visibility": row.get("timestamp_visibility_category", ""),
            "position": row.get("timestamp_candidate_position", ""),
            "color": row.get("timestamp_color", ""),
            "time_of_day": row.get("estimated_time_of_day", ""),
            "quality_tags": row.get("quality_tags", ""),
            "image_found": image_found,
            "crop_written": crop_written,
            "annotated_written": annotated_written,
            "original_path": str(out_original) if image_found else "",
            "crop_path": str(out_crop) if crop_written else "",
            "annotated_path": str(out_annotated) if annotated_written else "",
        })

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "problem_report.csv", index=False, encoding="utf-8-sig")
    print(f"Exported {len(report)} problem rows to {out_dir}")
    print(report["problem"].value_counts().to_string())


if __name__ == "__main__":
    main()
