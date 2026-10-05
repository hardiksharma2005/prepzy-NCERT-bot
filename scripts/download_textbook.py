"""Download the 13 chapters of the current NCERT Class 10 Science textbook."""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
BASE_URL = "https://ncert.nic.in/textbook/pdf/{code}.pdf"


def download(code: str, attempts: int = 5) -> None:
    target = PDF_DIR / f"{code}.pdf"
    if target.exists() and target.stat().st_size > 100_000:
        print(f"{code}: already present")
        return
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(BASE_URL.format(code=code), timeout=120,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            target.write_bytes(resp.content)
            print(f"{code}: {len(resp.content) // 1024} KB")
            return
        except requests.RequestException as exc:
            print(f"{code}: attempt {attempt} failed ({exc})")
            time.sleep(3 * attempt)
    sys.exit(f"Could not download {code}")


if __name__ == "__main__":
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    chapters = json.loads((ROOT / "data" / "chapters.json").read_text(encoding="utf-8"))
    for code in chapters:
        download(code)
