"""Download NASA's raw JSON archive without loading upstream pickle objects."""
import argparse
from datetime import datetime, timezone
import subprocess
import time
import urllib.request
import zipfile
from common import ROOT, DATA_URL, save_json, sha256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    archive = ROOT / "data/raw/airfoil-learning-dataset.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or args.force:
        partial = archive.with_suffix(".zip.part")
        start = time.monotonic()
        with urllib.request.urlopen(DATA_URL, timeout=120) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            done, last = 0, 0
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > 20:
                    print(f"Downloaded {done/1e6:.0f}/{total/1e6:.0f} MB ({done/1e6/max(1,time.monotonic()-start):.1f} MB/s)", flush=True)
                    last = time.monotonic()
            if total and done != total:
                raise ValueError("Incomplete download; the final archive has not been replaced.")
        partial.replace(archive)
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if not entries or not all(i.filename.startswith("json/") and i.filename.endswith(".json") for i in entries):
            raise ValueError("Unexpected NASA archive schema.")
    source = ROOT / "data/raw/source.json"
    digest = sha256(archive)
    if source.exists() and not args.force:
        import json
        if json.loads(source.read_text())["sha256"] != digest:
            raise ValueError("Archive differs from the saved checksum.")
    else:
        upstream = ROOT / "upstream/airfoil-learning"
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() if (upstream / ".git").exists() else None
        save_json(source, {"repository": "https://github.com/nasa/airfoil-learning", "revision": revision,
            "dataset_url": DATA_URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": digest, "bytes": archive.stat().st_size, "json_files": len(entries),
            "license_reference": "NASA repository README: NASA Open Source Agreement; upstream files retained under upstream/airfoil-learning",
            "labels": "Source XFOIL results; no new CFD or experimental measurements"})
    print(f"Verified {len(entries)} JSON files; SHA-256 {digest}")

if __name__ == "__main__":
    main()
