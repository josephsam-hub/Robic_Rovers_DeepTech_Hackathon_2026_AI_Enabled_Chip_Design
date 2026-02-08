import os, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- CONFIG ----------------
CLASS_PATHS = {
    "Clean": "/content/drive/MyDrive/IESA-Deep-Tech/Datasets/clean",
    "Bridge": "/content/drive/MyDrive/bridge/images",
    "Crack": "/content/drive/MyDrive/IESA-Deep-Tech/Datasets/Crack Defective",
    "LER": "/content/drive/MyDrive/IESA-Deep-Tech/Datasets/LER Defect",
    "LineCollapse": "/content/drive/MyDrive/Line collapse/images",
    "Open": "/content/drive/MyDrive/open/images",
    "Scratch": "/content/drive/MyDrive/IESA-Deep-Tech/Datasets/Scratches Defect",
    "Via": "/content/drive/MyDrive/IESA-Deep-Tech/Datasets/Via Defect"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 40

SAVE_DIR = "/content/drive/MyDrive/Models Version-4"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_PTH = f"{SAVE_DIR}/best_wafer_model.pth"
INT8_PT = f"{SAVE_DIR}/wafer_final_int8.pt"

# ---------------- TRANSFORMS ----------------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- DATASET ----------------
class WaferDataset(Dataset):
    def __init__(self, paths, transform):
        self.imgs, self.labels = [], []
        self.cls2idx = {c:i for i,c in enumerate(paths.keys())}
        for c,p in paths.items():
            for f in os.listdir(p):
                if f.lower().endswith(('.jpg','.png','.jpeg')):
                    self.imgs.append(os.path.join(p,f))
                    self.labels.append(self.cls2idx[c])
        self.transform = transform

    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = datasets.folder.default_loader(self.imgs[idx])
        return self.transform(img), self.labels[idx]

ds = WaferDataset(CLASS_PATHS, train_tf)
train_len = int(0.85 * len(ds))
train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, len(ds)-train_len])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = models.squeezenet1_1(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
for p in model.features.parameters():
    p.requires_grad = False

model.to(DEVICE)

optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# ---------------- TRAIN ----------------
best_acc = 0
print(f"🚀 Training on {len(ds)} images")

for epoch in range(EPOCHS):
    model.train()

    if epoch == 10:
        print("🔓 Unfreezing backbone")
        for p in model.features.parameters():
            p.requires_grad = True
        optimizer.add_param_group({'params': model.features.parameters(), 'lr': 1e-5})

    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for x,y in val_loader:
            out = model(x.to(DEVICE))
            preds.extend(out.argmax(1).cpu())
            labs.extend(y)

    acc = accuracy_score(labs, preds)
    scheduler.step(acc)

    print(f"Epoch {epoch+1}: Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BEST_PTH)

# ---------------- EXPORT INT8 ----------------
print("⚙️ Exporting INT8 TorchScript")

model.load_state_dict(torch.load(BEST_PTH))
model.eval().cpu()

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        prune.l1_unstructured(m, "weight", 0.2)
        prune.remove(m, "weight")

quant = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, torch.qint8)
scripted = torch.jit.trace(quant, torch.randn(1,3,256,256))
scripted.save(INT8_PT)

print("✅ Saved:")
print(BEST_PTH)
print(INT8_PT)
