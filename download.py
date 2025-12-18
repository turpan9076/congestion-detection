import os
import time
from datetime import datetime, timedelta
import requests

BASE_URL = "https://www.seishiga.kkr.mlit.go.jp/himeji/pic"

exclude = {455, 458, 459, 462, 474, 476, 480, 482, 484, 485, 488, 494}

camera_urls = {
    f"C{num:05d}": f"{BASE_URL}/C{num:05d}.jpg"
    for num in range(452, 496)
    if num not in exclude
}

BASE_DIR = "./input"
TARGET_MINUTES = {0, 10, 20, 30, 40, 50}


def download_image(cam_id, url, retries=3):
    save_dir = os.path.join(BASE_DIR, cam_id)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dst_path = os.path.join(save_dir, f"{cam_id}_{timestamp}.jpg")

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                f.write(r.content)
            print(f"[{cam_id}] Downloaded: {dst_path}")
            return True
        except Exception as e:
            print(f"[{cam_id}] Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    print(f"[{cam_id}] Failed after {retries} attempts")
    return False


def download_images():
    for cam_id, url in camera_urls.items():
        download_image(cam_id, url)


def wait_until_next_target():
    now = datetime.now()
    next_minute = ((now.minute // 10) + 1) * 10
    if next_minute >= 60:
        next_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)
    sleep_time = (next_time - now).total_seconds()
    print(f"Waiting until {next_time.strftime('%H:%M:%S')} ({sleep_time:.0f}s)")
    time.sleep(sleep_time)


if __name__ == "__main__":
    while True:
        now = datetime.now()
        if now.minute in TARGET_MINUTES and now.second < 5:
            print(f"\n=== Download start at {now.strftime('%Y-%m-%d %H:%M:%S')} ===")
            try:
                download_images()
            except Exception as e:
                print(f"Unexpected error: {e}")
            print("Download finished.\n")
            wait_until_next_target()
        else:
            wait_until_next_target()