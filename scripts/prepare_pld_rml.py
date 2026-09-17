"""Audit and extract the already-downloaded PLD 4x RML/GT2RML ZIPs. No downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


def prepare(root, extract=False, smoke_zip=None):
    root = Path(root).expanduser().resolve()
    sources = [("4x Downsampled RML", "4x_rml", 1),
               ("4x Downsampled GT2RML", "4x_undistorted_GT2RML", 2)]
    inventories = []
    total_bytes = 0
    for source, destination, camera in sources:
        members = {}
        for archive in sorted((root / "4x_downsampled" / source).glob("*.zip")):
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    if not info.filename.lower().endswith((".tif", ".tiff")) or info.filename.startswith("__MACOSX"):
                        continue
                    name = Path(info.filename).name
                    match = re.fullmatch(r"(?:warped_4x_undistorted|4x)_img_(\d+)_cam_(\d+)\.tiff", name)
                    if not match or int(match[2]) != camera:
                        raise ValueError(f"Unexpected image member: {info.filename}")
                    image_id = int(match[1])
                    if image_id in members:
                        raise ValueError(f"Duplicate ID {image_id}: {archive}")
                    members[image_id] = (archive, info.filename, name, info.file_size, info.CRC)
                    total_bytes += info.file_size
        if set(members) != set(range(100000)):
            raise ValueError(f"Expected IDs 0..99999 in {source}; found {len(members)}")
        inventories.append((destination, members))
    print(json.dumps({"pairs": 100000, "uncompressed_gib": total_bytes / 2**30,
                      "free_gib": shutil.disk_usage(root).free / 2**30}), flush=True)
    if not extract:
        return
    missing_bytes = sum(item[3] for folder, members in inventories for item in members.values()
                        if not (root / folder / item[2]).exists())
    if shutil.disk_usage(root).free < missing_bytes + 20 * 2**30:
        raise RuntimeError("Insufficient space for extraction plus 20 GiB reserve")
    smoke_ids = {1000, 2000, 3000, 4000, 5000, 25000, 50000, 75000}
    smoke = zipfile.ZipFile(smoke_zip, "w", zipfile.ZIP_DEFLATED) if smoke_zip else None
    try:
        digest = hashlib.sha256()
        for folder, members in inventories:
            out = root / folder
            out.mkdir(exist_ok=True)
            grouped = {}
            for image_id, item in sorted(members.items()):
                grouped.setdefault(item[0], []).append((image_id, item))
                digest.update(f"{folder}/{item[2]}:{item[3]}:{item[4]}\n".encode())
            for archive, items in grouped.items():
                with zipfile.ZipFile(archive) as z:
                    for image_id, (_, member, name, size, crc) in items:
                        path = out / name  # Strict filename regex prevents path traversal.
                        if path.exists():
                            import zlib
                            if path.stat().st_size != size or zlib.crc32(path.read_bytes()) != crc:
                                raise ValueError(f"Existing extracted file failed CRC verification: {path}")
                        else:
                            temporary = path.with_suffix(".tiff.part")
                            with z.open(member) as source, temporary.open("wb") as target:
                                shutil.copyfileobj(source, target)
                            temporary.replace(path)  # ZIP reader verifies CRC before replacement.
                        if smoke and image_id in smoke_ids:
                            smoke.write(path, f"{folder}/{name}")
                print(f"Extracted and checked {archive.name}: {len(items)} files", flush=True)
        manifest = {"dataset": "PLD 4x RML", "pairs": 100000, "shape": [300, 480, 3],
                    "train": {"first_id": 5000, "last_id": 99999, "count": 95000},
                    "validation": {"first_id": 1000, "last_id": 4999, "count": 4000},
                    "test": {"first_id": 0, "last_id": 999, "count": 1000},
                    "inventory_sha256": digest.hexdigest(), "uncompressed_bytes": total_bytes,
                    "verification": "ZIP CRC on extraction; existing files checked against member CRC"}
        (root / "rml_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)
    finally:
        if smoke:
            smoke.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--smoke_zip")
    args = parser.parse_args()
    prepare(args.root, args.extract, args.smoke_zip)
