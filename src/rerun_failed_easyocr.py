import argparse
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
from tqdm import tqdm

from postprocess_timestamp_results import choose_candidate


ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-/:. "


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess_variants(crop):
    variants = []
    h, w = crop.shape[:2]
    padded = cv2.copyMakeBorder(crop, 18, 18, 26, 26, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    variants.append(("gray4", cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)))
    variants.append(("gray6", cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))
    variants.append(("raw4", cv2.resize(padded, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)))

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe4", cv2.resize(clahe, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)))
    clahe_strong = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(gray)
    variants.append(("clahe_strong6", cv2.resize(clahe_strong, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    lab_l, lab_a, lab_b = cv2.split(lab)
    lab_l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(lab_l)
    lab_clahe = cv2.cvtColor(cv2.merge([lab_l, lab_a, lab_b]), cv2.COLOR_LAB2BGR)
    variants.append(("lab_clahe5", cv2.resize(lab_clahe, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)))

    local_bg = cv2.GaussianBlur(gray, (0, 0), 9)
    local_contrast = cv2.addWeighted(gray, 1.8, local_bg, -0.8, 24)
    variants.append(("local_contrast6", cv2.resize(local_contrast, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    inv = cv2.bitwise_not(gray)
    variants.append(("inv4", cv2.resize(inv, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    variants.append(("sharp4", cv2.resize(sharp, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)))

    for thresh_name, thresh_type in [("otsu", cv2.THRESH_BINARY), ("otsu_inv", cv2.THRESH_BINARY_INV)]:
        _, th = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        variants.append((f"{thresh_name}4", cv2.resize(th, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)))

    # White OSD over bright backgrounds often disappears in grayscale.
    # Morphological contrast/edge variants make the strokes separable again.
    for k in [9, 15, 21, 31]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, k // 3)))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        variants.append((f"blackhat{k}_6", cv2.resize(blackhat, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))
        variants.append((f"tophat{k}_6", cv2.resize(tophat, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))
        variants.append((f"grad{k}_6", cv2.resize(grad, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))
        variants.append((f"tophat{k}_inv6", cv2.resize(cv2.bitwise_not(tophat), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    edges = cv2.Canny(gray, 40, 140)
    variants.append(("canny6", cv2.resize(edges, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    adaptive_inv = cv2.bitwise_not(adaptive)
    variants.append(("adaptive6", cv2.resize(adaptive, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))
    variants.append(("adaptive_inv6", cv2.resize(adaptive_inv, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    for block, c in [(15, -2), (21, -4), (41, 2)]:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c)
        variants.append((f"adaptive_mean{block}_{c}_6", cv2.resize(th, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))
        variants.append((f"adaptive_mean{block}_{c}_inv6", cv2.resize(cv2.bitwise_not(th), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    lower_white = np.array([180, 180, 180], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(padded, lower_white, upper_white)
    variants.append(("white_mask6", cv2.resize(white_mask, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    kernel_erode = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(gray, kernel_erode, iterations=1)
    variants.append(("eroded6", cv2.resize(eroded, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    _, eroded_otsu = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("eroded_otsu6", cv2.resize(eroded_otsu, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    kernel_dilate = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(gray, kernel_dilate, iterations=1)
    variants.append(("dilated6", cv2.resize(dilated, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    gamma = 0.4
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, table)
    variants.append(("gamma04_6", cv2.resize(gamma_corrected, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)))

    if w > 0 and h > 0:
        wide = cv2.resize(padded, (max(256, int(w * 5)), max(48, int(h * 5))), interpolation=cv2.INTER_CUBIC)
        variants.append(("wide5", wide))
    return variants


def fallback_bbox_for_position(image, position):
    h, w = image.shape[:2]
    pos = str(position or "").lower()
    crop_w = int(w * 0.45)
    crop_h = max(70, int(h * 0.12))
    if "right" in pos:
        x1, x2 = max(0, w - crop_w), w
    else:
        x1, x2 = 0, min(w, crop_w)
    if "bottom" in pos:
        y1, y2 = max(0, h - crop_h), h
    else:
        y1, y2 = 0, min(h, crop_h)
    return x1, y1, x2, y2


def candidate_bboxes(image, bbox, position):
    h, w = image.shape[:2]
    boxes = []
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        pad_x = max(18, int(box_w * 0.06))
        pad_y = max(22, int(box_h * 0.60))
        boxes.append(("detected_pad", max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)))

        # If YOLO clipped a left/right timestamp edge, a wider horizontal box often recovers the date.
        wide_x1 = 0 if x1 < w // 2 else max(0, int(w * 0.52))
        wide_x2 = min(w, int(w * 0.48)) if x1 < w // 2 else w
        wide_y1 = max(0, y1 - max(28, int(box_h * 0.9)))
        wide_y2 = min(h, y2 + max(28, int(box_h * 0.9)))
        boxes.append(("detected_side_wide", wide_x1, wide_y1, wide_x2, wide_y2))

    fx1, fy1, fx2, fy2 = fallback_bbox_for_position(image, position)
    boxes.append(("position_fallback", fx1, fy1, fx2, fy2))

    pos = str(position or "").lower()
    strip_h = max(90, int(h * 0.16))
    if "bottom" in pos:
        sy1, sy2 = max(0, h - strip_h), h
    else:
        sy1, sy2 = 0, min(h, strip_h)
    if "right" in pos:
        sx1, sx2 = max(0, int(w * 0.45)), w
    else:
        sx1, sx2 = 0, min(w, int(w * 0.55))
    boxes.append(("position_strip", sx1, sy1, sx2, sy2))

    deduped = []
    seen = set()
    for name, x1, y1, x2, y2 in boxes:
        key = (x1, y1, x2, y2)
        if key not in seen and x2 > x1 and y2 > y1:
            seen.add(key)
            deduped.append((name, x1, y1, x2, y2))
    return deduped


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
        x1, y1, x2, y2 = [int(float(part)) for part in str(value).split(",")]
        return x1, y1, x2, y2
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Re-run EasyOCR on failed rows with stronger preprocessing")
    parser.add_argument("--in-csv", default="results12_postprocessed.csv")
    parser.add_argument("--out-csv", default="results12_enhanced.csv")
    parser.add_argument("--images-dir", default="selected")
    parser.add_argument("--target-statuses", default="PARSE_FAILED,NEED_REVIEW,TIME_WRONG")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0, help="Process only first N target rows; 0 means all")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--fast", action="store_true", help="Use only the fastest preprocessing variants")
    parser.add_argument("--allow-reference-repair", action="store_true", help="Use reference date to repair OCR-damaged date fields")
    parser.add_argument("--variant-set", default="focused", choices=["focused", "all"], help="Preprocessing variant set")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    for col in ["ocr_text_ready_model", "parsed_time", "status_reason", "comment"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)
    targets = {s.strip() for s in args.target_statuses.split(",") if s.strip()}
    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    changed = 0
    target_indices = [idx for idx, row in df.iterrows() if row.get("status") in targets]
    if args.max_rows > 0:
        target_indices = target_indices[: args.max_rows]
    for idx in tqdm(target_indices, desc="rerun failed OCR"):
        row = df.loc[idx]
        image_path = find_image(args.images_dir, row.get("image_path", ""), row.get("split", ""))
        bbox = parse_bbox(row.get("detected_bbox", ""))
        if image_path is None:
            continue
        image = imread(image_path)
        if image is None:
            continue
        best = None
        focused_names = {
            "gray6",
            "clahe_strong6",
            "lab_clahe5",
            "local_contrast6",
            "sharp4",
            "otsu4",
            "otsu_inv4",
            "tophat21_6",
            "tophat31_6",
            "tophat31_inv6",
            "adaptive6",
            "adaptive_inv6",
            "adaptive_mean15_-2_6",
            "adaptive_mean15_-2_inv6",
            "adaptive_mean21_-4_6",
            "adaptive_mean21_-4_inv6",
            "wide5",
            "white_mask6",
            "eroded6",
            "eroded_otsu6",
            "dilated6",
            "gamma04_6",
        }
        for crop_name, x1, y1, x2, y2 in candidate_bboxes(image, bbox, row.get("timestamp_candidate_position", "")):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            variants = preprocess_variants(crop)
            if args.fast:
                variants = variants[:4]
            elif args.variant_set == "focused":
                variants = [(name, image_variant) for name, image_variant in variants if name in focused_names]
            for method, variant in variants:
                texts = reader.readtext(variant, detail=0, paragraph=False, allowlist=ALLOWLIST)
                raw_text = " ".join(texts).strip()
                if not raw_text:
                    continue
                try:
                    dt, delta, status, reason = choose_candidate(
                        raw_text,
                        row.get("reference_time", ""),
                        row.get("timezone", "") or "Asia/Yekaterinburg",
                        args.tolerance,
                        allow_reference_repair=args.allow_reference_repair,
                    )
                except Exception:
                    continue
                if dt is None:
                    continue
                score = 0 if status == "OK" else 1 if status == "TIMEZONE_MISMATCH" else 2 if status == "TIME_WRONG" else 3
                candidate = (score, delta, status, dt, raw_text, f"{crop_name}/{method}", reason)
                if best is None or candidate < best:
                    best = candidate
                    if status == "OK":
                        break
            if best is not None and best[2] == "OK":
                break

        if best is None:
            continue
        _, delta, status, dt, raw_text, method, reason = best
        old_status = row.get("status")
        if status in {"OK", "TIMEZONE_MISMATCH"} or old_status == "PARSE_FAILED":
            changed += 1
            df.at[idx, "ocr_text_ready_model"] = raw_text
            df.at[idx, "parsed_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            df.at[idx, "delta_minutes"] = delta
            df.at[idx, "status"] = status
            df.at[idx, "status_reason"] = f"enhanced_ocr from {old_status}; method={method}; {reason}"
            df.at[idx, "comment"] = f"Enhanced method: {method}"

        if args.save_every > 0 and changed > 0 and changed % args.save_every == 0:
            df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Changed rows: {changed}")
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
