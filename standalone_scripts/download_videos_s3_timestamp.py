import boto3
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time
from threading import Lock

BUCKET      = "marmonwwkyqj9tw60bsfhvc-training"
PREFIX      = "Station 1/Videos"
REGION      = "us-east-2"
OUTPUT_DIR  = r"D:\code\RPN-DinoV2\datasets\marmon_station_1\videos"
MAX_WORKERS = 8    # concurrent downloads, s3-sync style (each also multiparts internally)

# Date range — inclusive, filename-based (YYYY-MM-DD in "YYYY-MM-DD HH-MM-SS.mkv")
DATE_START  = datetime(2026, 8, 6).date()
DATE_END    = datetime(2026, 8, 8).date()

# Daily time-of-day window — inclusive
DAY_START   = time(0, 0, 0)
DAY_END     = time(23, 59, 59)

# Of the files matching the date/time window, only download the N largest (by S3
# object size) per day -- largest tends to track the most motion/activity in frame.
TOP_N_PER_DAY = 30


def _download_one(s3, key, local_path, filename, size):
    print(f"  [dl]   {filename}  ({size / 1e6:.1f} MB)")
    s3.download_file(BUCKET, key, local_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")

    skipped    = 0
    already    = 0
    bad_name   = 0
    by_day     = defaultdict(list)  # date -> [(key, filename, size), ...]

    print(f"Scanning s3://{BUCKET}/{PREFIX}")
    print(f"Date range: {DATE_START} -> {DATE_END}")
    print(f"Time window: {DAY_START} -> {DAY_END}")
    print(f"Top {TOP_N_PER_DAY} largest per day\n")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = os.path.basename(key)

            if not filename.lower().endswith((".mkv", ".mp4")):
                skipped += 1
                continue

            try:
                file_dt = datetime.strptime(filename[:-4], "%Y-%m-%d %H-%M-%S")
            except ValueError:
                bad_name += 1
                continue

            if not (DATE_START <= file_dt.date() <= DATE_END):
                skipped += 1
                continue

            if not (DAY_START <= file_dt.time() <= DAY_END):
                skipped += 1
                continue

            by_day[file_dt.date()].append((key, filename, obj["Size"]))

    to_download = []  # (key, local_path, filename, size)
    for day in sorted(by_day):
        candidates = by_day[day]
        skipped += max(len(candidates) - TOP_N_PER_DAY, 0)
        top = sorted(candidates, key=lambda c: -c[2])[:TOP_N_PER_DAY]
        print(f"  {day}: {len(candidates)} match, keeping {len(top)} largest "
              f"({top[-1][2] / 1e6:.1f}-{top[0][2] / 1e6:.1f} MB)")

        for key, filename, size in top:
            local_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(local_path) and os.path.getsize(local_path) == size:
                print(f"    [skip] {filename} (already exists, size matches)")
                already += 1
                continue
            to_download.append((key, local_path, filename, size))

    # Parallel downloads (s3-sync style) — each client call also multiparts
    # internally via boto3's default transfer config, so this layers file-level
    # concurrency on top of that.
    downloaded = 0
    failed     = 0
    lock = Lock()
    print(f"\n{len(to_download)} file(s) to download, {MAX_WORKERS} at a time ...\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_download_one, s3, key, local_path, filename, size): filename
            for key, local_path, filename, size in to_download
        }
        for fut in as_completed(futures):
            filename = futures[fut]
            try:
                fut.result()
                with lock:
                    downloaded += 1
            except Exception as e:
                print(f"  [FAIL] {filename}: {e}")
                with lock:
                    failed += 1

    print(f"\nDone. Downloaded={downloaded}  Failed={failed}  AlreadyExisted={already}  "
          f"BadName={bad_name}  Skipped(no match)={skipped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
