# civitai_fetch_resume.py
# Robust Civitai downloader with resume + dedupe + partial saves + graceful Ctrl+C.

import os
import time
import random
import json
from pathlib import Path

import httpx
import pandas as pd

# ========= SETTINGS (you can tweak) =========
PERIOD         = "AllTime"         # "AllTime", "Year", "Month", "Week", "Day"
SORT           = "Most Reactions"  # or "Newest"
NSFW           = "false"           # keep SFW only

LIMIT          = 100               # smaller pages help avoid 429s
MAX_PAGES      = 200               # pages THIS RUN (you can run again later)
REQUEST_DELAY  = 4.0               # seconds between pages
JITTER_RANGE   = (0.5, 1.5)        # small random wait to avoid bursts
MAX_RETRIES    = 8                 # retries per request on 429/5xx
SAVE_EVERY     = 1                 # save partial + checkpoint after EVERY page

# Output files (keep names stable so resume/merge works)
OUT_CSV        = "civitai_images_alltime.csv"          # open in Excel
OUT_PARQUET    = "civitai_images_alltime.parquet"
OUT_PARTIAL    = "civitai_images_alltime_partial.parquet"
CHECKPOINT     = "civitai_checkpoint_alltime.json"

# Optional: API token (NOT required for public browsing)
TOKEN = os.getenv("CIVITAI_TOKEN")

BASE_URL = "https://civitai.com/api/v1/images"
BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}


def get_with_backoff(client: httpx.Client, url: str, params=None, headers=None, max_retries: int = 8) -> httpx.Response:
    """GET with exponential backoff on 429/5xx; honors Retry-After if present."""
    hdrs = dict(BASE_HEADERS)
    if headers:
        hdrs.update(headers)
    if TOKEN:
        hdrs["Authorization"] = f"Bearer {TOKEN}"

    attempt = 0
    while True:
        try:
            r = client.get(url, params=params, headers=hdrs, timeout=60)

            # If rate-limited or server hiccup, back off and retry
            if r.status_code == 429 or (500 <= r.status_code < 600):
                attempt += 1
                if attempt > max_retries:
                    r.raise_for_status()

                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = int(retry_after)
                else:
                    sleep_s = min(120, 2 ** attempt)  # 2,4,8,16,32,64,120 cap
                print(f"[{r.status_code}] backing off {sleep_s}s …")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r

        except httpx.HTTPError as e:
            # Network error or non-2xx after retries above
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(120, 2 ** attempt)
            print(f"[HTTP error] retry {attempt}/{max_retries} in {sleep_s}s: {e}")
            time.sleep(sleep_s)


def load_checkpoint() -> str | None:
    """Return next_url if checkpoint exists, else None."""
    if not os.path.exists(CHECKPOINT):
        return None
    try:
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("next_url")
    except Exception:
        return None


def save_checkpoint(next_url: str | None) -> None:
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"next_url": next_url, "ts": time.time()}, f)


def load_existing_ids() -> set[str]:
    """
    Read existing dataset (parquet > partial > csv) and return set of image_ids
    so we can skip duplicates.
    """
    for p in (OUT_PARQUET, OUT_PARTIAL, OUT_CSV):
        if os.path.exists(p):
            try:
                if p.endswith(".parquet"):
                    df_existing = pd.read_parquet(p, columns=["image_id"])
                else:
                    df_existing = pd.read_csv(p, usecols=["image_id"])
                return set(df_existing["image_id"].astype(str))
            except Exception:
                continue
    return set()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric reaction cols & compute total_reactions."""
    cols = ["like_count", "heart_count", "comment_count", "laugh_count", "cry_count"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df[cols] = df[cols].fillna(0).astype("int64")
    df["total_reactions"] = df[cols].sum(axis="columns")
    return df


def main() -> None:
    Path(".").mkdir(parents=True, exist_ok=True)

    # RESUME: either start with base URL or with next_url from checkpoint
    next_url = load_checkpoint()
    if next_url:
        print(f"Resuming from checkpoint: {next_url}")
        url = next_url
        params = None
    else:
        url = BASE_URL
        params = {"limit": LIMIT, "sort": SORT, "period": PERIOD, "nsfw": NSFW}

    # Skip duplicates we already have
    existing_ids = load_existing_ids()
    if existing_ids:
        print(f"Found {len(existing_ids)} existing image IDs. Will skip duplicates.")

    rows: list[dict] = []
    pages_done = 0
    last_next_url: str | None = None

    try:
        with httpx.Client() as client:
            while url and pages_done < MAX_PAGES:
                print(f"Fetching page {pages_done + 1}/{MAX_PAGES} …")
                r = get_with_backoff(
                    client,
                    url,
                    params=params if url == BASE_URL else None,
                    max_retries=MAX_RETRIES,
                )
                data = r.json()

                items = data.get("items", []) or []
                if not items:
                    print("No more items returned. Stopping early.")
                    break

                new_count = 0
                for it in items:
                    iid = str(it.get("id"))
                    if iid in existing_ids:
                        continue  # already have it
                    stats = it.get("stats", {}) or {}
                    meta = it.get("meta", {}) or {}
                    rows.append({
                        "platform": "civitai",
                        "image_id": iid,
                        "image_url": it.get("url"),
                        "width": it.get("width"),
                        "height": it.get("height"),
                        "created_at": it.get("createdAt"),
                        "post_id": it.get("postId"),
                        "nsfw": it.get("nsfw"),
                        "nsfwLevel": it.get("nsfwLevel"),
                        "like_count": stats.get("likeCount", 0),
                        "heart_count": stats.get("heartCount", 0),
                        "comment_count": stats.get("commentCount", 0),
                        "laugh_count": stats.get("laughCount", 0),
                        "cry_count": stats.get("cryCount", 0),
                        "prompt": meta.get("prompt") or meta.get("Prompt"),
                        "negative_prompt": meta.get("negativePrompt"),
                        "model": meta.get("Model"),
                        "username": it.get("username"),
                    })
                    existing_ids.add(iid)
                    new_count += 1

                pages_done += 1
                print(f"  +{new_count} new rows this page (total this run: {len(rows)})")

                # cursor to next page
                last_next_url = (data.get("metadata") or {}).get("nextPage")
                url = last_next_url
                params = None  # from now on, follow the provided URL

                # partial save + checkpoint every page (SAVE_EVERY = 1)
                if pages_done % SAVE_EVERY == 0:
                    if rows:
                        df_part = normalize(pd.DataFrame(rows))
                        df_part.to_parquet(OUT_PARTIAL, index=False)
                        print(f"Partial saved ({len(df_part)} rows) → {OUT_PARTIAL}")
                    save_checkpoint(last_next_url)
                    print(f"Checkpoint saved with next_url = {last_next_url!r}")

                # polite delay with jitter
                time.sleep(REQUEST_DELAY + random.uniform(*JITTER_RANGE))

    except KeyboardInterrupt:
        print("\nStopping… (Ctrl+C)")
        # Save whatever we have + the latest cursor
        if rows:
            df_part = normalize(pd.DataFrame(rows))
            df_part.to_parquet(OUT_PARTIAL, index=False)
            print(f"Partial saved on interrupt ({len(df_part)} rows) → {OUT_PARTIAL}")
        save_checkpoint(last_next_url)
        print(f"Checkpoint saved with next_url = {last_next_url!r}")

    # Final merge/save (runs whether finished or interrupted)
    if not rows:
        print("No new data this run. Nothing to merge.")
        return

    new_df = normalize(pd.DataFrame(rows))

    # Merge with existing final file if present, drop dupes by image_id
    if os.path.exists(OUT_PARQUET):
        try:
            prev = pd.read_parquet(OUT_PARQUET)
        except Exception:
            prev = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else None
        if prev is not None:
            combined = pd.concat([prev, new_df], ignore_index=True)
        else:
            combined = new_df
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["image_id"], keep="first")

    print(f"Saving FINAL dataset: {len(combined)} rows → {OUT_CSV} / {OUT_PARQUET}")
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8")
    try:
        combined.to_parquet(OUT_PARQUET, index=False)
    except Exception as e:
        print(f"Parquet save failed (ok for now): {e}")

    # Tiny preview
    print(combined.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
