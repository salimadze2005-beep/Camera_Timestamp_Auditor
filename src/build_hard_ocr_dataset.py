import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytz


GOOD_STATUSES = {"OK", "TIMEZONE_MISMATCH"}


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
        x1, y1, x2, y2 = [int(float(part)) for part in str(value).split(",")]
        return x1, y1, x2, y2
    except Exception:
        return None


def label_from_reference(ref_time, tz_name):
    ref_dt = pd.to_datetime(ref_time)
    if ref_dt.tzinfo is None:
        ref_dt = pytz.utc.localize(ref_dt)
    local = ref_dt.astimezone(pytz.timezone(tz_name or "Asia/Yekaterinburg"))
    return local.strftime("%Y-%m-%d %H:%M:%S")


def crop_with_padding(image, bbox, pad_x=14, pad_y=10):
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def augment(crop):
    img = crop.copy()

    if random.random() < 0.8:
        scale = random.uniform(1.2, 4.5)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if random.random() < 0.55:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if random.random() < 0.5:
            img = gray
        else:
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if random.random() < 0.35:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 4.0), tileGridSize=(8, 8))
        img = clahe.apply(gray)
        if random.random() < 0.4:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if random.random() < 0.25:
        img = cv2.bitwise_not(img)

    if random.random() < 0.55:
        alpha = random.uniform(0.65, 1.55)
        beta = random.uniform(-35, 35)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.45:
        noise = np.random.normal(0, random.uniform(2, 18), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.45:
        k = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.25:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

    if random.random() < 0.35:
        angle = random.uniform(-3.0, 3.0)
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.35:
        quality = random.randint(30, 90)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img


def main():
    parser = argparse.ArgumentParser(description="Build hard-case OCR dataset from failed timestamp rows")
    parser.add_argument("--results-csv", default="ensemble_best_results.csv")
    parser.add_argument("--images-dir", default="selected")
    parser.add_argument("--out-dir", default="hard_ocr_data")
    parser.add_argument("--augmentations-per-crop", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.results_csv)
    hard = df[~df["status"].isin(GOOD_STATUSES)].copy()
    hard = hard[hard["detected_bbox"].notna()]

    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_lines = []
    val_lines = []
    used = 0
    skipped = 0

    for _, row in hard.iterrows():
        image_path = find_image(args.images_dir, row.get("image_path", ""), row.get("split", "train"))
        bbox = parse_bbox(row.get("detected_bbox", ""))
        if image_path is None or bbox is None:
            skipped += 1
            continue
        image = imread(image_path)
        if image is None:
            skipped += 1
            continue
        crop = crop_with_padding(image, bbox)
        if crop is None:
            skipped += 1
            continue

        label = label_from_reference(row.get("reference_time", ""), row.get("timezone", "Asia/Yekaterinburg"))
        split = "val" if row.get("split") == "val" else "train"
        target_dir = val_dir if split == "val" else train_dir
        lines = val_lines if split == "val" else train_lines
        stem = Path(str(row.get("image_path", image_path))).stem

        base_name = f"{stem}_hard_base.jpg"
        imwrite(target_dir / base_name, crop)
        lines.append(f"{split}/{base_name}\t{label}\n")

        for idx in range(args.augmentations_per_crop):
            aug = augment(crop)
            aug_name = f"{stem}_hard_aug_{idx:03d}.jpg"
            imwrite(target_dir / aug_name, aug)
            lines.append(f"{split}/{aug_name}\t{label}\n")
        used += 1

    (out_dir / "train.txt").write_text("".join(train_lines), encoding="utf-8")
    (out_dir / "val.txt").write_text("".join(val_lines), encoding="utf-8")
    chars = sorted(set("".join(line.split("\t", 1)[1].strip() for line in train_lines + val_lines)))
    (out_dir / "dict.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")

    print(f"Hard rows used: {used}; skipped: {skipped}")
    print(f"Train samples: {len(train_lines)}")
    print(f"Val samples: {len(val_lines)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
