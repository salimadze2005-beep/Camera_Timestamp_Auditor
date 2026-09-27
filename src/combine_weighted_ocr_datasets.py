import argparse
import shutil
from pathlib import Path


def read_lines(dataset_dir, split):
    path = Path(dataset_dir) / f"{split}.txt"
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_dataset(src_dir, out_dir, split, name, repeat):
    lines_out = []
    target_dir = out_dir / split
    target_dir.mkdir(parents=True, exist_ok=True)
    for line in read_lines(src_dir, split):
        rel_path, label = line.split("\t", 1)
        src_img = Path(src_dir) / rel_path
        if not src_img.exists():
            continue
        for rep in range(repeat):
            dst_name = f"{name}_r{rep}_{Path(rel_path).name}"
            dst_img = target_dir / dst_name
            if not dst_img.exists():
                shutil.copyfile(src_img, dst_img)
            lines_out.append(f"{split}/{dst_name}\t{label}\n")
    return lines_out


def main():
    parser = argparse.ArgumentParser(description="Combine OCR datasets with repeat weights")
    parser.add_argument("--out-dir", default="train_ocr_data_hardmix")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec in name=path:repeat form, e.g. real=train_ocr_data:1 hard=hard_ocr_data:5",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for item in args.dataset:
        name_path, repeat_str = item.rsplit(":", 1)
        name, path = name_path.split("=", 1)
        specs.append((name, Path(path), int(repeat_str)))

    for split in ["train", "val"]:
        lines = []
        for name, path, repeat in specs:
            lines.extend(copy_dataset(path, out_dir, split, name, repeat))
        (out_dir / f"{split}.txt").write_text("".join(lines), encoding="utf-8")
        print(f"{split}: {len(lines)} samples")

    chars = set()
    for split in ["train", "val"]:
        for line in read_lines(out_dir, split):
            _, label = line.split("\t", 1)
            chars.update(label)
    (out_dir / "dict.txt").write_text("\n".join(sorted(chars)) + "\n", encoding="utf-8")
    print(f"Combined weighted dataset written to {out_dir}")
    print("Characters:", "".join(sorted(chars)))


if __name__ == "__main__":
    main()
