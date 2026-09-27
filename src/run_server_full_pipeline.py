import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd, cwd):
    print("\n>>>", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def summarize(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    counts = df["status"].value_counts()
    total = len(df)
    ok = int((df["status"] == "OK").sum())
    ok_tz = int(df["status"].isin(["OK", "TIMEZONE_MISMATCH"]).sum())
    return {
        "csv": str(path),
        "total": total,
        "OK": ok,
        "OK_pct": ok / total * 100 if total else 0,
        "OK_or_TIMEZONE_MISMATCH": ok_tz,
        "OK_or_TIMEZONE_MISMATCH_pct": ok_tz / total * 100 if total else 0,
        "counts": counts.to_dict(),
    }


def print_summary(items):
    print("\n================ SUMMARY ================")
    for item in items:
        if not item:
            continue
        print(f"\n{item['csv']}")
        print(f"OK: {item['OK']}/{item['total']} ({item['OK_pct']:.2f}%)")
        print(
            "OK+TIMEZONE_MISMATCH: "
            f"{item['OK_or_TIMEZONE_MISMATCH']}/{item['total']} "
            f"({item['OK_or_TIMEZONE_MISMATCH_pct']:.2f}%)"
        )
        for status, count in item["counts"].items():
            print(f"  {status}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Full GPU/server pipeline sweep for camera timestamp OCR")
    parser.add_argument("--project-dir", default=".", help="Path to easyOCR_project")
    parser.add_argument("--images-dir", default="selected")
    parser.add_argument("--manifest", default="selected_manifest.csv")
    parser.add_argument("--yolo-model", default="detector_for_annotation.pt")
    parser.add_argument("--baseline-csv", default="results12.csv")
    parser.add_argument("--timezone-default", default="Asia/Yekaterinburg")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR rerun and EasyOCR Reader")

    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--synthetic-dir", default="train_ocr_data_synthetic")
    parser.add_argument("--synthetic-train-count", type=int, default=30000)
    parser.add_argument("--synthetic-val-count", type=int, default=5000)
    parser.add_argument("--combined-dir", default="train_ocr_data_combined")
    parser.add_argument("--real-ocr-dir", default="train_ocr_data")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--trained-model", default="best_easyocr_server.pth")
    parser.add_argument("--pretrained-weights", default="")
    parser.add_argument("--resume-model", default="", help="Resume OCR training from this checkpoint")

    parser.add_argument("--run-trained-inference", action="store_true")
    parser.add_argument("--run-enhanced-rerun", action="store_true")
    parser.add_argument("--target-statuses", default="PARSE_FAILED,NEED_REVIEW,TIME_WRONG")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    py = sys.executable

    outputs = []

    baseline_post = project_dir / "results12_postprocessed_server.csv"
    run(
        [
            py,
            "postprocess_timestamp_results.py",
            "--in-csv",
            args.baseline_csv,
            "--out-csv",
            str(baseline_post.name),
            "--tolerance",
            str(args.tolerance),
            "--timezone-default",
            args.timezone_default,
        ],
        project_dir,
    )
    outputs.append(summarize(baseline_post))

    if args.run_enhanced_rerun:
        enhanced = project_dir / "results12_enhanced_server.csv"
        cmd = [
            py,
            "rerun_failed_easyocr.py",
            "--in-csv",
            str(baseline_post.name),
            "--out-csv",
            str(enhanced.name),
            "--images-dir",
            args.images_dir,
            "--target-statuses",
            args.target_statuses,
            "--tolerance",
            str(args.tolerance),
            "--save-every",
            "10",
        ]
        if args.gpu:
            cmd.append("--gpu")
        run(cmd, project_dir)
        outputs.append(summarize(enhanced))

    if args.generate_synthetic:
        run(
            [
                py,
                "generate_synthetic_ocr_data.py",
                "--out-dir",
                args.synthetic_dir,
                "--train-count",
                str(args.synthetic_train_count),
                "--val-count",
                str(args.synthetic_val_count),
            ],
            project_dir,
        )

    if args.train or args.run_trained_inference:
        run(
            [
                py,
                "combine_ocr_datasets.py",
                "--real-dir",
                args.real_ocr_dir,
                "--synthetic-dir",
                args.synthetic_dir,
                "--out-dir",
                args.combined_dir,
            ],
            project_dir,
        )

    if args.train:
        train_cmd = [
            py,
            "train_easyocr.py",
            "--data-dir",
            args.combined_dir,
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--save-model",
            args.trained_model,
        ]
        if args.pretrained_weights:
            train_cmd.extend(["--pretrained-weights", args.pretrained_weights])
        if args.resume_model:
            train_cmd.extend(["--resume-model", args.resume_model])
        run(train_cmd, project_dir)

    if args.run_trained_inference:
        trained_raw = project_dir / "ocr_easyocr_server_results.csv"
        run(
            [
                py,
                "run_custom_easyocr.py",
                "--manifest",
                args.manifest,
                "--images-dir",
                args.images_dir,
                "--out",
                str(trained_raw.name),
                "--yolo-model",
                args.yolo_model,
                "--ocr-model",
                args.trained_model,
                "--timezone-default",
                args.timezone_default,
                "--tolerance",
                str(args.tolerance),
                "--failed-crops-dir",
                "failed_crops_server_trained",
            ],
            project_dir,
        )
        outputs.append(summarize(trained_raw))

        trained_post = project_dir / "ocr_easyocr_server_results_postprocessed.csv"
        run(
            [
                py,
                "postprocess_timestamp_results.py",
                "--in-csv",
                str(trained_raw.name),
                "--out-csv",
                str(trained_post.name),
                "--tolerance",
                str(args.tolerance),
                "--timezone-default",
                args.timezone_default,
            ],
            project_dir,
        )
        outputs.append(summarize(trained_post))

    print_summary(outputs)


if __name__ == "__main__":
    main()
