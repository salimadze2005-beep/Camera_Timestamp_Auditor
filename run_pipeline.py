from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera Timestamp Auditor")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--yolo-model", default="", help="Необязательные кастомные YOLO-веса")
    parser.add_argument("--skip-enhanced", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    baseline = out / "baseline.csv"
    cmd = [
        py, "src/run_easyocr.py",
        "--images-dir", args.images_dir,
        "--manifest", args.manifest,
        "--out", str(baseline),
        "--tolerance", str(args.tolerance),
    ]
    if args.gpu:
        cmd.append("--gpu")
    if args.yolo_model:
        cmd.extend(["--yolo-model", args.yolo_model])
    run(cmd, root)

    post = out / "baseline_postprocessed.csv"
    run([
        py, "src/postprocess_timestamp_results.py",
        "--in-csv", str(baseline),
        "--out-csv", str(post),
        "--tolerance", str(args.tolerance),
    ], root)

    inputs = [f"baseline={post}"]
    if not args.skip_enhanced:
        enhanced = out / "enhanced.csv"
        cmd = [
            py, "src/rerun_failed_easyocr.py",
            "--in-csv", str(post),
            "--out-csv", str(enhanced),
            "--images-dir", args.images_dir,
            "--tolerance", str(args.tolerance),
            "--target-statuses", "PARSE_FAILED,NEED_REVIEW,TIME_WRONG,OCR_FAILED,TIMESTAMP_NOT_FOUND",
            "--variant-set", "focused",
        ]
        if args.gpu:
            cmd.append("--gpu")
        run(cmd, root)
        inputs.insert(0, f"enhanced={enhanced}")

    ensemble = out / "ensemble_result.csv"
    run([py, "src/make_ocr_ensemble.py", "--out", str(ensemble), "--inputs", *inputs], root)
    print(f"\nГотово: {ensemble}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
