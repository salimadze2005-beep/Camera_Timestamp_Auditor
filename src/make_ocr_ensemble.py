import argparse
from pathlib import Path

import pandas as pd


STATUS_RANK = {
    "OK": 0,
    "TIMEZONE_MISMATCH": 1,
    "TIME_WRONG": 2,
    "NEED_REVIEW": 3,
    "PARSE_FAILED": 4,
    "OCR_FAILED": 4,
    "TIMESTAMP_NOT_FOUND": 5,
    "REFERENCE_TIME_MISSING": 6,
}


def delta_value(row):
    value = row.get("delta_minutes", "")
    try:
        if pd.isna(value) or str(value).strip() == "":
            return 10**12
        return float(value)
    except Exception:
        return 10**12


def main():
    parser = argparse.ArgumentParser(description="Build best-row ensemble from several OCR result CSVs")
    parser.add_argument("--out", default="ensemble_best_results.csv")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV inputs in name=path form, for example enhanced=results12_enhanced.csv trained=ocr_trained.csv",
    )
    args = parser.parse_args()

    named_frames = []
    for item in args.inputs:
        if "=" not in item:
            raise ValueError(f"Input must be name=path, got: {item}")
        name, path = item.split("=", 1)
        df = pd.read_csv(path)
        df["_ensemble_source"] = name
        named_frames.append((name, Path(path), df))

    if not named_frames:
        raise ValueError("No input CSVs were provided")

    lengths = {len(df) for _, _, df in named_frames}
    if len(lengths) != 1:
        raise ValueError(f"All CSVs must have the same row count, got lengths: {sorted(lengths)}")

    chosen_rows = []
    for idx in range(next(iter(lengths))):
        options = [df.iloc[idx].copy() for _, _, df in named_frames]
        options.sort(
            key=lambda row: (
                STATUS_RANK.get(str(row.get("status", "")), 99),
                delta_value(row),
            )
        )
        chosen = options[0]
        source = chosen.pop("_ensemble_source")
        old_comment = "" if pd.isna(chosen.get("comment", "")) else str(chosen.get("comment", ""))
        chosen["comment"] = (old_comment + f" | ensemble_source={source}").strip()
        chosen_rows.append(chosen)

    out = pd.DataFrame(chosen_rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Saved ensemble to {args.out}")
    print(out["status"].value_counts().to_string())
    total = len(out)
    ok = int((out["status"] == "OK").sum())
    ok_tz = int(out["status"].isin(["OK", "TIMEZONE_MISMATCH"]).sum())
    print(f"OK: {ok}/{total} ({ok / total:.2%})")
    print(f"OK+TIMEZONE_MISMATCH: {ok_tz}/{total} ({ok_tz / total:.2%})")
    print("\nSources:")
    print(out["comment"].str.extract(r"ensemble_source=([^ |]+)")[0].value_counts().to_string())


if __name__ == "__main__":
    main()
