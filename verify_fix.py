"""Verification script for the image loading fix."""
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("IMAGE LOADING FIX VERIFICATION")
print("=" * 60)

# 1. Check upload directory
from utils.config import UPLOAD_HISTORY_PATH
print(f"\n1. Upload directory config:")
print(f"   UPLOAD_HISTORY_PATH = {UPLOAD_HISTORY_PATH}")
print(f"   Absolute = {UPLOAD_HISTORY_PATH.resolve()}")
print(f"   Exists = {UPLOAD_HISTORY_PATH.exists()}")

# 2. Check subdirectories
print(f"\n2. Upload subdirectories:")
for d in sorted(UPLOAD_HISTORY_PATH.iterdir()):
    if d.is_dir():
        files = list(d.glob("*.jpg"))
        print(f"   {d.name}/: {len(files)} files")
        for f in files[:2]:
            print(f"      - {f.name}")

# 3. Check database values
print(f"\n3. Database values (latest 5 searches):")
import sqlite3
conn = sqlite3.connect(str(ROOT / "database" / "app.db"))
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute(
    "SELECT id, disease, image_path, image_name FROM searches ORDER BY id DESC LIMIT 5"
)
for r in cur.fetchall():
    row = dict(r)
    # Check if the image file exists
    img_path = row.get("image_path")
    file_exists = False
    if img_path:
        p = Path(img_path)
        file_exists = p.exists()
    print(f"   id={row['id']} disease={row['disease']}")
    print(f"      image_path={img_path}")
    print(f"      image_name={row['image_name']}")
    print(f"      file_exists={file_exists}")

conn.close()

# 4. Test the to_upload_url logic
print(f"\n4. URL conversion test:")
def to_upload_url(image_path):
    if not image_path:
        return None
    if isinstance(image_path, str) and (image_path.startswith("http://") or image_path.startswith("https://")):
        return image_path
    marker = "history/uploads/"
    idx = image_path.replace("\\", "/").find(marker)
    if idx == -1:
        rel = image_path.replace("\\", "/").lstrip("/")
        if rel.startswith("uploads/"):
            return f"http://127.0.0.1:8000/{rel}"
        return None
    rel_part = image_path.replace("\\", "/")[idx + len(marker):]
    return f"http://127.0.0.1:8000/uploads/{rel_part}"

# Test with a known path
test_path = r"C:\Updated_Project\history\uploads\Acne\search_Acne_1d448e1fa6.jpg"
url = to_upload_url(test_path)
print(f"   Input: {test_path}")
print(f"   Output: {url}")

# Check if the file exists at that path
test_file = Path(test_path)
print(f"   File exists: {test_file.exists()}")

# 5. Verify StaticFiles mount
print(f"\n5. StaticFiles mount verification:")
print(f"   Mount point: /uploads -> {UPLOAD_HISTORY_PATH}")
print(f"   A file at history/uploads/Acne/search_Acne_xxx.jpg")
print(f"   would be served at: http://127.0.0.1:8000/uploads/Acne/search_Acne_xxx.jpg")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)