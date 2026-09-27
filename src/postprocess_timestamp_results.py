import argparse
import calendar
import re
from datetime import datetime

import pandas as pd
import pytz


WEEKDAYS = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "WEN": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}

OCR_REPLACEMENTS = {
    "O": "0",
    "I": "1",
    "L": "1",
    "|": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "Q": "0",
}


def normalize_ocr_text(raw_text):
    text = "" if pd.isna(raw_text) else str(raw_text).upper()
    text = re.sub(r"\b(WEN|WED|MON|TUE|THU|FRI|SAT|SUN)\b", " ", text)
    text = text.replace("AM", " AM ").replace("PM", " PM ")
    for old, new in OCR_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def valid_datetime(year, month, day, hour, minute, second):
    try:
        year, month, day = int(year), int(month), int(day)
        hour, minute, second = int(hour), int(minute), int(second)
        if year < 2000 or year > 2030:
            return None
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def apply_ampm(hour, text):
    hour = int(hour)
    upper = text.upper()
    if "PM" in upper and hour < 12:
        return hour + 12
    if "AM" in upper and hour == 12:
        return 0
    return hour


def add_candidate(candidates, text, year, month, day, hour, minute, second="00", source=""):
    hour = apply_ampm(hour, text)
    dt = valid_datetime(year, month, day, hour, minute, second)
    if dt is not None:
        candidates.append((dt, source))


def extract_weekday(raw_text):
    text = "" if pd.isna(raw_text) else str(raw_text).upper()
    for token, idx in WEEKDAYS.items():
        if re.search(rf"\b{token}\b", text):
            return token, idx
    return None, None


def iter_digit_windows(digits, min_len=10, max_len=14):
    for size in range(max_len, min_len - 1, -1):
        if len(digits) < size:
            continue
        for start in range(0, len(digits) - size + 1):
            yield digits[start : start + size]


def candidates_from_window(text, window):
    out = []
    formats = []
    if len(window) >= 12:
        formats.extend(
            [
                ("YMDHMS", window[0:4], window[4:6], window[6:8], window[8:10], window[10:12], window[12:14] or "00"),
                ("DMYHMS", window[4:8], window[2:4], window[0:2], window[8:10], window[10:12], window[12:14] or "00"),
                ("MDYHMS", window[4:8], window[0:2], window[2:4], window[8:10], window[10:12], window[12:14] or "00"),
            ]
        )
    if len(window) >= 10:
        formats.extend(
            [
                ("YMDHM", window[0:4], window[4:6], window[6:8], window[8:10], window[10:12], "00"),
                ("DMYHM", window[4:8], window[2:4], window[0:2], window[8:10], window[10:12], "00"),
                ("MDYHM", window[4:8], window[0:2], window[2:4], window[8:10], window[10:12], "00"),
            ]
        )
    for source, year, month, day, hour, minute, second in formats:
        add_candidate(out, text, year, month, day, hour, minute, second, source)
    return out


def extract_datetime_candidates(raw_text):
    text = normalize_ocr_text(raw_text)
    candidates = []

    patterns = [
        (r"(20[0-3]\d)\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})\D{0,5}(\d{1,2})\D{0,3}(\d{2})\D{0,3}(\d{2})", "YMD"),
        (r"(\d{1,2})\D{0,3}(\d{1,2})\D{0,3}(20[0-3]\d)\D{0,5}(\d{1,2})\D{0,3}(\d{2})\D{0,3}(\d{2})", "DMY_OR_MDY"),
        (r"(20[0-3]\d)\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})\D{0,5}(\d{1,2})\D{0,3}(\d{2})", "YMD_NOSEC"),
        (r"(\d{1,2})\D{0,3}(\d{1,2})\D{0,3}(20[0-3]\d)\D{0,5}(\d{1,2})\D{0,3}(\d{2})", "DMY_OR_MDY_NOSEC"),
    ]

    for pattern, kind in patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            if kind == "YMD":
                add_candidate(candidates, text, *groups, source=kind)
            elif kind == "YMD_NOSEC":
                add_candidate(candidates, text, *groups, "00", source=kind)
            elif kind == "DMY_OR_MDY":
                a, b, year, hour, minute, second = groups
                add_candidate(candidates, text, year, b, a, hour, minute, second, "DMY")
                add_candidate(candidates, text, year, a, b, hour, minute, second, "MDY")
            elif kind == "DMY_OR_MDY_NOSEC":
                a, b, year, hour, minute = groups
                add_candidate(candidates, text, year, b, a, hour, minute, "00", "DMY_NOSEC")
                add_candidate(candidates, text, year, a, b, hour, minute, "00", "MDY_NOSEC")

    digits = re.sub(r"\D", "", text)
    for window in iter_digit_windows(digits):
        candidates.extend(candidates_from_window(text, window))

    deduped = {}
    for dt, source in candidates:
        deduped.setdefault(dt, source)
    return [(dt, source) for dt, source in deduped.items()]


def reference_to_local(ref_time, tz_name):
    ref_dt = pd.to_datetime(ref_time)
    if ref_dt.tzinfo is None:
        ref_dt = pytz.utc.localize(ref_dt)
    return ref_dt.astimezone(pytz.timezone(tz_name)).replace(tzinfo=None)


def classify_delta(delta_minutes, tolerance):
    if delta_minutes <= tolerance:
        return "OK"
    if delta_minutes > 1300:
        return "NEED_REVIEW"
    mod_60 = delta_minutes % 60
    if mod_60 <= tolerance or mod_60 >= (60 - tolerance):
        return "TIMEZONE_MISMATCH"
    return "TIME_WRONG"


def add_ref_date_time_candidates(out, ref_options, hour, minute, second, source):
    for ref_dt in ref_options:
        dt = valid_datetime(ref_dt.year, ref_dt.month, ref_dt.day, hour, minute, second)
        if dt is not None:
            out.append((dt, source))


def reference_repair_candidates(raw_text, ref_time, tz_name):
    text = normalize_ocr_text(raw_text)
    digits = re.sub(r"\D", "", text)
    if not ref_time:
        return []

    ref_local = reference_to_local(ref_time, tz_name)
    out = []
    year_matches = list(re.finditer(r"20[0-3]\d", digits))
    date_options = [ref_local]
    date_options.extend([ref_local + pd.Timedelta(days=1), ref_local - pd.Timedelta(days=1)])

    for match in year_matches:
        tail = digits[match.end():]
        time_windows = []
        for size in [6, 5, 4]:
            if len(tail) >= size:
                for start in range(0, len(tail) - size + 1):
                    time_windows.append(tail[start:start + size])
        if len(digits) >= 6:
            time_windows.extend([digits[-6:], digits[-5:], digits[-4:]])

        for tw in dict.fromkeys(time_windows):
            pieces = []
            if len(tw) >= 6:
                pieces.append((tw[0:2], tw[2:4], tw[4:6]))
                pieces.append((tw[-6:-4], tw[-4:-2], tw[-2:]))
            if len(tw) == 5:
                pieces.append((tw[0:2], tw[2:4], f"0{tw[4]}"))
                pieces.append((tw[0:2], f"0{tw[2]}", tw[3:5]))
                pieces.append((f"0{tw[0]}", tw[1:3], tw[3:5]))
            if len(tw) == 4:
                pieces.append((tw[0:2], tw[2:4], "00"))
            for hour, minute, second in pieces:
                for ref_dt in date_options:
                    dt = valid_datetime(ref_dt.year, ref_dt.month, ref_dt.day, hour, minute, second)
                    if dt is not None:
                        out.append((dt, "REF_REPAIR_TIME_TAIL"))

    # If the date is unreadable but the time survived, use the known capture date.
    # This handles washed-out OSDs like "... 16.27:04" where the left date is OCR noise.
    time_patterns = [
        r"(?<!\d)([0-2]?\d)\D{1,3}([0-5]\d)\D{1,3}([0-5]\d)(?!\d)",
        r"(?<!\d)([0-2]?\d)\D{1,3}([0-5]\d)(?!\d)",
    ]
    for pattern in time_patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            hour, minute = groups[0], groups[1]
            second = groups[2] if len(groups) > 2 else "00"
            add_ref_date_time_candidates(out, date_options, hour, minute, second, "REF_REPAIR_TIME_ONLY")

    # Date without a reliable year: try both DD/MM and MM/DD using the reference year.
    partial_patterns = [
        r"(?<!\d)(\d{1,2})\D{1,3}(\d{1,2})\D{1,6}([0-2]?\d)\D{1,3}([0-5]\d)\D{1,3}([0-5]\d)(?!\d)",
        r"(?<!\d)(\d{1,2})\D{1,3}(\d{1,2})\D{1,6}([0-2]?\d)\D{1,3}([0-5]\d)(?!\d)",
    ]
    for pattern in partial_patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            a, b, hour, minute = groups[0], groups[1], groups[2], groups[3]
            second = groups[4] if len(groups) > 4 else "00"
            for ref_dt in date_options:
                add_candidate(out, text, ref_dt.year, b, a, hour, minute, second, "REF_REPAIR_DMY_PARTIAL")
                add_candidate(out, text, ref_dt.year, a, b, hour, minute, second, "REF_REPAIR_MDY_PARTIAL")

    # Compact OCR fragments often preserve the last HHMMSS/HMMSS/MMSS digits.
    for tw in dict.fromkeys([digits[-6:], digits[-5:], digits[-4:]]):
        if len(tw) < 4:
            continue
        pieces = []
        if len(tw) == 6:
            pieces.append((tw[0:2], tw[2:4], tw[4:6]))
        elif len(tw) == 5:
            pieces.append((tw[0:1], tw[1:3], tw[3:5]))
            pieces.append((tw[0:2], tw[2:4], f"0{tw[4]}"))
        elif len(tw) == 4:
            pieces.append((tw[0:2], tw[2:4], "00"))
        for hour, minute, second in pieces:
            add_ref_date_time_candidates(out, date_options, hour, minute, second, "REF_REPAIR_COMPACT_TIME")

    deduped = {}
    for dt, source in out:
        deduped.setdefault(dt, source)
    return [(dt, source) for dt, source in deduped.items()]


def choose_candidate(raw_text, ref_time, tz_name, tolerance, allow_reference_repair=False):
    candidates = extract_datetime_candidates(raw_text)
    if allow_reference_repair:
        candidates.extend(reference_repair_candidates(raw_text, ref_time, tz_name))
    if not candidates:
        return None, None, None, "no_candidates"

    ref_local = reference_to_local(ref_time, tz_name)
    weekday_token, weekday_idx = extract_weekday(raw_text)

    ranked = []
    for dt, source in candidates:
        delta = abs((dt - ref_local).total_seconds()) / 60.0
        weekday_penalty = 0
        if weekday_idx is not None and dt.weekday() != weekday_idx:
            weekday_penalty = 720
        ranked.append((delta + weekday_penalty, delta, dt, source))

    ranked.sort(key=lambda item: item[0])
    _, delta, dt, source = ranked[0]
    status = classify_delta(round(delta, 2), tolerance)
    reason = f"candidate_source={source}"
    if weekday_token:
        reason += f"; weekday={weekday_token}/{calendar.day_abbr[dt.weekday()].upper()}"
    return dt, round(delta, 2), status, reason


def main():
    parser = argparse.ArgumentParser(description="Reference-aware timestamp postprocessor for EasyOCR results")
    parser.add_argument("--in-csv", default="results12.csv")
    parser.add_argument("--out-csv", default="results12_postprocessed.csv")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--timezone-default", default="Asia/Yekaterinburg")
    parser.add_argument("--only-statuses", default="NEED_REVIEW,TIME_WRONG,PARSE_FAILED")
    parser.add_argument("--allow-reference-repair", action="store_true", help="Use reference date to repair OCR-damaged date fields")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    target_statuses = {s.strip() for s in args.only_statuses.split(",") if s.strip()}

    changed = 0
    for idx, row in df.iterrows():
        if row.get("status") not in target_statuses:
            continue
        raw_text = row.get("ocr_text_ready_model", "")
        if pd.isna(raw_text) or not str(raw_text).strip():
            raw_text = row.get("ocr_text_trained_model", "")
        if pd.isna(raw_text) or not str(raw_text).strip():
            continue
        try:
            dt, delta, status, reason = choose_candidate(
                raw_text,
                row.get("reference_time", ""),
                row.get("timezone", "") or args.timezone_default,
                args.tolerance,
                allow_reference_repair=args.allow_reference_repair,
            )
        except Exception as exc:
            df.at[idx, "status_reason"] = f"postprocess_error={exc}"
            continue
        if dt is None:
            continue
        old_status = row.get("status")
        old_parsed = row.get("parsed_time", "")
        new_parsed = dt.strftime("%Y-%m-%d %H:%M:%S")
        if new_parsed != old_parsed or status != old_status:
            changed += 1
            df.at[idx, "parsed_time"] = new_parsed
            df.at[idx, "delta_minutes"] = delta
            df.at[idx, "status"] = status
            df.at[idx, "status_reason"] = f"postprocessed from {old_status}; {reason}"

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Changed rows: {changed}")
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
