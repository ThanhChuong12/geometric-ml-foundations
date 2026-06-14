import os, sys, numpy as np, torch, torch.nn as nn, torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.pointnet_model import PointNetFull

CLASSES = ['airplane', 'chair', 'car', 'lamp', 'table']
NUM_POINTS = 1024
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'data', 'sample_clouds')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'models')

def augment(pts):
    angles = np.random.uniform(0, 2*np.pi, 3)
    Rx = np.array([[1,0,0],[0,np.cos(angles[0]),-np.sin(angles[0])],[0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],[0,1,0],[-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],[np.sin(angles[2]),np.cos(angles[2]),0],[0,0,1]])
    pts = pts @ (Rz @ Ry @ Rx).T
    pts = pts * np.random.uniform(0.85, 1.15)
    pts = pts + np.random.randn(*pts.shape) * 0.02
    idx = np.random.choice(pts.shape[0], NUM_POINTS, replace=(pts.shape[0]<NUM_POINTS))
    pts = pts[idx]; pts -= pts.mean(axis=0)
    d = np.max(np.linalg.norm(pts, axis=1))
    if d > 0: pts /= d
    return pts.astype(np.float32)

class DS(torch.utils.data.Dataset):
    def __init__(self, spc, aug):
        self.aug, self.data = aug, []
        for label, cls in enumerate(CLASSES):
            pts = np.load(os.path.join(DATA_DIR, f"{cls}_sample.npy"))
            for _ in range(spc): self.data.append((pts, label))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        pts, label = self.data[idx]
        if self.aug:
            pts = augment(pts)
        else:
            N = pts.shape[0]
            i = np.random.choice(N, NUM_POINTS, replace=(N<NUM_POINTS))
            pts = pts[i]; pts -= pts.mean(axis=0)
            d = np.max(np.linalg.norm(pts, axis=1))
            if d > 0: pts /= d
            pts = pts.astype(np.float32)
        return torch.from_numpy(pts), label

if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    tl = torch.utils.data.DataLoader(DS(100, True), batch_size=8, shuffle=True)
    vl = torch.utils.data.DataLoader(DS(30, False), batch_size=8, shuffle=False)

    model = PointNetFull(num_classes=5)
    # Freeze CA HAI T-Net de tranh mode collapse tren small synthetic dataset
    # Input T-Net (3x3) va Feature T-Net (64x64) deu bi freeze
    for name, param in model.named_parameters():
        if 'transform' in name:  # input_transform + feature_transform
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    save_path = os.path.join(MODELS_DIR, 'pointnet_cls.pth')

    print("Training PointNet Full (feature_transform FROZEN)")
    for epoch in range(1, 41):
        model.train()
        total_loss = correct = total = 0
        for pts, labels in tl:
            optimizer.zero_grad()
            logits, _, _ = model(pts)
            loss = criterion(logits, labels)  # NO reg loss
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1)==labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 40:
            model.eval()
            vc = vt = 0
            with torch.no_grad():
                for pts, labels in vl:
                    logits, _, _ = model(pts)
                    vc += (logits.argmax(1)==labels).sum().item()
                    vt += labels.size(0)
            va = vc/vt*100
            if va > best_acc:
                best_acc = va
                torch.save(model.state_dict(), save_path)
                tag = " <- SAVED"
            else: tag = ""
            print(f"  Epoch {epoch:3d}/40 | Loss {total_loss/len(tl):.4f} | Train {correct/total*100:.1f}% | Val {va:.1f}%{tag}")
        else:
            print(f"  Epoch {epoch:3d}/40 | Loss {total_loss/len(tl):.4f} | Train {correct/total*100:.1f}%")

    print(f"\nDONE! Best: {best_acc:.1f}%")
    
    # Quick verify
    model.eval()
    model.load_state_dict(torch.load(save_path))
    for cls in CLASSES:
        pts = np.load(os.path.join(DATA_DIR, f"{cls}_sample.npy"))
        N = pts.shape[0]
        i = np.random.choice(N, NUM_POINTS, replace=(N<NUM_POINTS))
        pts = pts[i].astype(np.float32)
        pts -= pts.mean(axis=0); d = np.max(np.linalg.norm(pts, axis=1))
        if d > 0: pts /= d
        t = torch.from_numpy(pts).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = model(t)
            pred = CLASSES[logits.argmax(1).item()]
            conf = torch.softmax(logits, 1).max().item()
        ok = 'OK' if pred == cls else 'WRONG'
        print(f"  {cls:10s} -> {pred:10s} {conf*100:.1f}% [{ok}]")
