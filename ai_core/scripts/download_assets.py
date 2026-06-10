"""
Script tải ModelNet40 HDF5 dataset và pretrained PointNet weights.

Chạy lệnh (từ thư mục demo/geo-ml-web/):
    python scripts/download_assets.py

Sẽ tải về:
    - backend/data/modelnet40_ply_hdf5_2048/ (~415MB) — ModelNet40 HDF5
    - backend/models/pointnet_cls.pth  — pretrained PointNet Full weights
    
Nếu download tự động thất bại, xem hướng dẫn manual bên dưới.
"""

import os
import sys
import urllib.request
import zipfile
import json
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(SCRIPT_DIR, '..', 'backend')
DATA_DIR     = os.path.join(BACKEND_DIR, 'data')
MODELS_DIR   = os.path.join(BACKEND_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)


# ─────────────────────────────────────────────────────────────
# Synthetic point clouds (fallback khi không download được data)
# ─────────────────────────────────────────────────────────────
def generate_synthetic_clouds():
    """
    Tạo point clouds giả cho 5 class demo (airplane, chair, car, lamp, table).
    Dùng để web chạy được ngay mà không cần download ModelNet40.
    Kết quả demo sẽ ít chính xác hơn data thực.
    """
    import numpy as np

    sample_dir = os.path.join(DATA_DIR, 'sample_clouds')
    os.makedirs(sample_dir, exist_ok=True)

    def _normalize(pts):
        pts -= pts.mean(axis=0)
        pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-6)
        return pts

    def make_airplane(n=2048):
        # Fuselage: cylinder along x-axis  (n//2 points)
        t = np.linspace(-1, 1, n // 2)
        r = 0.08 * (1 - 0.5 * t**2)
        angle = np.random.uniform(0, 2*np.pi, n // 2)
        fuselage = np.stack([t, r*np.cos(angle), r*np.sin(angle)], axis=1)
        # Main wings: flat rectangle  (n//4 points)
        wx = np.random.uniform(-0.3, 0.3, n // 4)
        wy = np.random.uniform(-0.7, 0.7, n // 4)
        wz = np.random.uniform(-0.02, 0.02, n // 4)
        wings = np.stack([wx, wy, wz], axis=1)
        # Tail: horizontal stabilizer near back  (n//8 points)
        tx = np.random.uniform(0.7, 0.95, n // 8)
        ty = np.random.uniform(-0.25, 0.25, n // 8)
        tz = np.random.uniform(-0.01, 0.01, n // 8)
        tail_h = np.stack([tx, ty, tz], axis=1)
        # Tail: vertical stabilizer  (n//8 points)
        vx = np.random.uniform(0.7, 0.95, n // 8)
        vy = np.random.uniform(-0.01, 0.01, n // 8)
        vz = np.random.uniform(0.0, 0.2, n // 8)
        tail_v = np.stack([vx, vy, vz], axis=1)
        pts = np.vstack([fuselage, wings, tail_h, tail_v])
        pts += np.random.randn(*pts.shape) * 0.01
        return _normalize(pts[:n])

    def make_chair(n=2048):
        # 4 legs
        legs = []
        for lx, ly in [(-0.4,-0.4),(0.4,-0.4),(-0.4,0.4),(0.4,0.4)]:
            z = np.random.uniform(-0.9, -0.1, n // 8)
            legs.append(np.stack([np.full_like(z, lx), np.full_like(z, ly), z], axis=1))
        # Seat: flat square
        sx = np.random.uniform(-0.5, 0.5, n // 4)
        sy = np.random.uniform(-0.5, 0.5, n // 4)
        sz = np.random.uniform(-0.1, 0.0, n // 4)
        seat = np.stack([sx, sy, sz], axis=1)
        # Backrest
        bx = np.random.uniform(-0.5, 0.5, n // 4)
        by = np.random.uniform(-0.4, -0.5, n // 4)
        bz = np.random.uniform(0.0, 0.9, n // 4)
        back = np.stack([bx, by, bz], axis=1)
        pts = np.vstack(legs + [seat, back])
        pts += np.random.randn(*pts.shape) * 0.01
        return _normalize(pts[:n])

    def make_car(n=2048):
        # Body: rounded box
        x = np.random.uniform(-0.8, 0.8, n * 2)
        y = np.random.uniform(-0.35, 0.35, n * 2)
        z = np.random.uniform(-0.2, 0.3, n * 2)
        mask = (np.abs(x) < 0.8) & (np.abs(y) < 0.35)
        pts = np.stack([x[mask], y[mask], z[mask]], axis=1)
        # Roof
        rx = np.random.uniform(-0.4, 0.4, n // 4)
        ry = np.random.uniform(-0.25, 0.25, n // 4)
        rz = np.random.uniform(0.3, 0.5, n // 4)
        roof = np.stack([rx, ry, rz], axis=1)
        pts = np.vstack([pts[:n*3//4], roof])
        pts += np.random.randn(*pts.shape) * 0.01
        return _normalize(pts[:n])

    def make_lamp(n=2048):
        # Pole: thin cylinder along z
        z = np.random.uniform(-0.8, 0.5, n // 2)
        r = 0.04
        angle = np.random.uniform(0, 2*np.pi, n // 2)
        pole = np.stack([r*np.cos(angle), r*np.sin(angle), z], axis=1)
        # Shade: cone at top
        t = np.random.uniform(0, 1, n // 2)
        rad = 0.05 + 0.25 * t
        angle2 = np.random.uniform(0, 2*np.pi, n // 2)
        shade = np.stack([rad*np.cos(angle2), rad*np.sin(angle2), 0.5 + 0.3*t], axis=1)
        pts = np.vstack([pole, shade])
        pts += np.random.randn(*pts.shape) * 0.01
        return _normalize(pts[:n])

    def make_table(n=2048):
        # Tabletop: flat rectangle
        tx = np.random.uniform(-0.8, 0.8, n // 2)
        ty = np.random.uniform(-0.5, 0.5, n // 2)
        tz = np.random.uniform(-0.03, 0.03, n // 2)
        top = np.stack([tx, ty, tz], axis=1)
        # 4 legs
        legs = []
        for lx, ly in [(-0.7,-0.4),(0.7,-0.4),(-0.7,0.4),(0.7,0.4)]:
            z = np.random.uniform(-0.8, -0.03, n // 8)
            legs.append(np.stack([np.full_like(z,lx), np.full_like(z,ly), z], axis=1))
        pts = np.vstack([top] + legs)
        pts += np.random.randn(*pts.shape) * 0.01
        return _normalize(pts[:n])

    generators = {
        'airplane': make_airplane,
        'chair':    make_chair,
        'car':      make_car,
        'lamp':     make_lamp,
        'table':    make_table,
    }

    print("[SYNTHETIC] Generating synthetic point clouds for demo...")
    for cls, gen_fn in generators.items():
        out_path = os.path.join(sample_dir, f"{cls}_sample.npy")
        pts = gen_fn(2048)
        np.save(out_path, pts.astype(np.float32))
        print(f"  Created: {cls}_sample.npy ({len(pts)} points)")
    print("[SYNTHETIC] Done.")


def save_class_names():
    """Lưu danh sách 40 class ModelNet40."""
    CLASSES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
        'bottle', 'bowl', 'car', 'chair', 'cone',
        'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
        'laptop', 'mantel', 'monitor', 'night_stand', 'person',
        'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent',
        'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    path = os.path.join(DATA_DIR, 'modelnet40_classes.json')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(CLASSES, f)
        print(f"  Saved: modelnet40_classes.json")


# ─────────────────────────────────────────────────────────────
# Download ModelNet40 HDF5 (thực)
# ─────────────────────────────────────────────────────────────
def download_modelnet40():
    """
    Tải ModelNet40 HDF5 từ nhiều mirror.
    Nếu tất cả đều fail → tạo synthetic data.
    """
    data_path = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')
    if os.path.exists(data_path) and any(f.endswith('.h5') for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))):
        print("[OK] ModelNet40 data already exists, skipping download.")
        return True

    mirrors = [
        "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
        "https://hkust-vgd.ust.hk/scanobjectnn/modelnet40_ply_hdf5_2048.zip",
        "https://storage.googleapis.com/pointnet-data/modelnet40_ply_hdf5_2048.zip",
    ]

    zip_path = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048.zip")
    success = False

    for i, url in enumerate(mirrors):
        print(f"[1/3] Trying mirror {i+1}: {url[:60]}...")
        try:
            urllib.request.urlretrieve(url, zip_path, show_progress)
            print()
            success = True
            break
        except Exception as e:
            print(f"\n  Failed: {e}")

    if not success:
        print("[WARNING] Could not download ModelNet40 from any mirror.")
        print("  -> Falling back to SYNTHETIC point clouds for demo.")
        print()
        print("  To get real ModelNet40 data, download manually:")
        print("  https://modelnet.cs.princeton.edu/  or")
        print("  https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip")
        print(f"  Then extract to: {data_path}")
        return False

    print("[1/3] Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"[1/3] Done. Data at: {data_path}")
    return True


def create_real_sample_npy_files():
    """Trích xuất file .npy mẫu từ HDF5 dataset thực."""
    import h5py

    sample_dir = os.path.join(DATA_DIR, 'sample_clouds')
    os.makedirs(sample_dir, exist_ok=True)

    CLASSES = json.load(open(os.path.join(DATA_DIR, 'modelnet40_classes.json')))
    DEMO_CLASSES = {'airplane': 0, 'chair': 8, 'car': 7, 'lamp': 19, 'table': 33}

    h5_dir  = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')
    txt_path = os.path.join(h5_dir, 'train_files.txt')

    if not os.path.exists(txt_path):
        print("[WARNING] train_files.txt not found. Skipping real sample extraction.")
        return

    with open(txt_path) as f:
        h5_files = [line.strip() for line in f]

    found = {c: False for c in DEMO_CLASSES}

    for h5_path in h5_files:
        full_path = os.path.join(h5_dir, os.path.basename(h5_path))
        if not os.path.exists(full_path):
            continue

        with h5py.File(full_path, 'r') as f:
            data   = f['data'][:]
            labels = f['label'][:].squeeze()

        for cls, cls_id in DEMO_CLASSES.items():
            if found[cls]:
                continue
            mask = (labels == cls_id)
            if not np.any(mask):
                continue
            idx = np.where(mask)[0][0]
            out = os.path.join(sample_dir, f"{cls}_sample.npy")
            np.save(out, data[idx].astype(np.float32))
            found[cls] = True
            print(f"  Saved real sample: {cls}_sample.npy")

        if all(found.values()):
            break

    print("[2/3] Sample extraction done.")


# ─────────────────────────────────────────────────────────────
# Download pretrained weights
# ─────────────────────────────────────────────────────────────
def download_pretrained_weights():
    """
    Tải pretrained PyTorch PointNet weights.
    Source: fxia22/pointnet.pytorch (accuracy ~89.2% on ModelNet40)
    """
    out_path = os.path.join(MODELS_DIR, 'pointnet_cls.pth')
    if os.path.exists(out_path):
        print("[OK] Pretrained weights already exist, skipping.")
        return True

    urls = [
        "https://github.com/fxia22/pointnet.pytorch/releases/download/v0.1/cls_model_249.pth",
    ]

    print("[3/3] Downloading pretrained PointNet weights...")
    for url in urls:
        try:
            urllib.request.urlretrieve(url, out_path, show_progress)
            print(f"\n[3/3] Weights saved to: {out_path}")
            return True
        except Exception as e:
            print(f"\n  Failed: {e}")

    print("[WARNING] Could not download pretrained weights.")
    print("  The model will use random weights (predictions will be random).")
    print("  Manual download: https://github.com/fxia22/pointnet.pytorch/releases")
    print(f"  Place file as: {out_path}")
    return False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  PointNet Asset Downloader")
    print("=" * 55)
    print()

    save_class_names()

    has_real_data = download_modelnet40()

    if has_real_data:
        create_real_sample_npy_files()
    else:
        generate_synthetic_clouds()

    download_pretrained_weights()

    print()
    print("=" * 55)
    print("  Setup complete. Start backend:")
    print("  cd backend && uvicorn main:app --reload")
    print("=" * 55)
