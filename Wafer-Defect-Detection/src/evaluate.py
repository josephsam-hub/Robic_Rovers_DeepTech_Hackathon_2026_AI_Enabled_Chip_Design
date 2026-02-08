import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files

# ---------------- CONFIG ----------------
CLASS_NAMES = [
    "Clean","Bridge","Crack","LER",
    "LineCollapse","Open","Scratch","Via"
]

CONF_THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/content/drive/MyDrive/Models Version-3/wafer_final_int8.pt"

# ---------------- LOAD MODEL ----------------
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# ---------------- PREPROCESS ----------------
tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- UPLOAD IMAGE ----------------
print("📤 Upload wafer image")
uploaded = files.upload()

for fname in uploaded:
    img = Image.open(fname).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x).view(-1)
        probs = F.softmax(logits, dim=0)

    conf, idx = torch.max(probs, 0)
    label = CLASS_NAMES[idx] if conf >= CONF_THRESH else "Uncertain"

    df = pd.DataFrame({
        "Defect": CLASS_NAMES,
        "Probability (%)": (probs.cpu().numpy()*100).round(2)
    }).sort_values("Probability (%)", ascending=False)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.imshow(img); plt.axis("off")
    plt.title(f"{label} ({conf.item()*100:.2f}%)")

    plt.subplot(1,2,2)
    plt.barh(df["Defect"], df["Probability (%)"])
    plt.gca().invert_yaxis()
    plt.show()

    print(df.to_string(index=False))
