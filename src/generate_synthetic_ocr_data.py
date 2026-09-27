import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np


FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M",
    "%m-%d-%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%a %Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S %a",
    "%a %d.%m.%Y %H:%M:%S",
]

COLORS = [
    ((245, 245, 245), (0, 0, 0)),
    ((20, 20, 20), (255, 255, 255)),
    ((210, 255, 80), (0, 0, 0)),
    ((255, 220, 80), (0, 0, 0)),
    ((180, 180, 180), (20, 20, 20)),
]


def render_text_image(label, width=420, height=64):
    fg, bg = random.choice(COLORS)
    image = np.full((height, width, 3), bg, dtype=np.uint8)

    font = random.choice(
        [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_PLAIN,
        ]
    )
    scale = random.uniform(0.72, 1.02)
    thickness = random.choice([1, 1, 2])
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    if tw > width - 12:
        scale *= (width - 12) / max(tw, 1)
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    x = random.randint(2, max(2, width - tw - 4))
    y = random.randint(th + 2, max(th + 2, height - baseline - 3))

    if random.random() < 0.45:
        shadow = tuple(int(c * 0.25) for c in fg)
        cv2.putText(image, label, (x + 1, y + 1), font, scale, shadow, thickness + 1, cv2.LINE_AA)
    cv2.putText(image, label, (x, y), font, scale, fg, thickness, cv2.LINE_AA)

    if random.random() < 0.65:
        alpha = random.uniform(0.65, 1.35)
        beta = random.uniform(-25, 25)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.55:
        noise = np.random.normal(0, random.uniform(2, 14), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.35:
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)

    if random.random() < 0.25:
        quality = random.randint(35, 85)
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if random.random() < 0.3:
        angle = random.uniform(-2.5, 2.5)
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return image


def make_label(dt):
    fmt = random.choice(FORMATS)
    label = dt.strftime(fmt).upper()
    if random.random() < 0.15:
        label = label.replace("WED", "WEN")
    return label


def write_split(out_dir, split, count, start_dt):
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx in range(count):
        dt = start_dt + timedelta(seconds=random.randint(0, 60 * 60 * 24 * 20))
        label = make_label(dt)
        width = random.randint(300, 560)
        height = random.randint(44, 84)
        image = render_text_image(label, width=width, height=height)
        filename = f"synthetic_{split}_{idx:06d}.jpg"
        cv2.imencode(".jpg", image)[1].tofile(str(split_dir / filename))
        lines.append(f"{split}/{filename}\t{label}\n")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OCR crops for camera timestamp recognizer")
    parser.add_argument("--out-dir", default="train_ocr_data_synthetic")
    parser.add_argument("--train-count", type=int, default=12000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_dt = datetime(2026, 6, 1, 0, 0, 0)

    train_lines = write_split(out_dir, "train", args.train_count, start_dt)
    val_lines = write_split(out_dir, "val", args.val_count, start_dt)

    (out_dir / "train.txt").write_text("".join(train_lines), encoding="utf-8")
    (out_dir / "val.txt").write_text("".join(val_lines), encoding="utf-8")

    chars = sorted(set("".join(line.split("\t", 1)[1].strip() for line in train_lines + val_lines)))
    (out_dir / "dict.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")

    print(f"Wrote {len(train_lines)} train and {len(val_lines)} val synthetic samples to {out_dir}")
    print("Characters:", "".join(chars))


if __name__ == "__main__":
    main()
