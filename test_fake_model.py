import os
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, models

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "fake_checkpoints/best_fake_detector.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\n🚀 Using Device: {DEVICE}")

# ============================================================
# IMAGE TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD MODEL
# ============================================================

print("\n🧠 Loading Model...\n")

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(

    nn.Dropout(0.4),

    nn.Linear(num_features, 256),

    nn.ReLU(),

    nn.BatchNorm1d(256),

    nn.Dropout(0.3),

    nn.Linear(256, 2)
)

# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("✅ Model Loaded Successfully!")

# ============================================================
# CLASS LABELS
# ============================================================

classes = [
    "FAKE AI-GENERATED IMAGE",
    "REAL HUMAN-CAPTURED IMAGE"
]

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    try:

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = Image.open(image_path).convert("RGB")

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        # ====================================================
        # EXTRACT CONFIDENCES
        # ====================================================

        fake_confidence = (
            probabilities[0][0].item() * 100
        )

        real_confidence = (
            probabilities[0][1].item() * 100
        )

        # ====================================================
        # THRESHOLD LOGIC
        # ====================================================

        if real_confidence >= 80:

            predicted_class = (
                "REAL HUMAN-CAPTURED IMAGE"
            )

            final_confidence = real_confidence

        elif real_confidence >= 60:

            predicted_class = (
                "⚠️ SUSPICIOUS / POSSIBLY AI-GENERATED"
            )

            final_confidence = real_confidence

        else:

            predicted_class = (
                "FAKE AI-GENERATED IMAGE"
            )

            final_confidence = fake_confidence

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print("\n========================================")

        print(
            f"\n🧠 Prediction: {predicted_class}"
        )

        print(
            f"🎯 Final Confidence: "
            f"{final_confidence:.2f}%"
        )

        print(
            f"\n📊 REAL Confidence: "
            f"{real_confidence:.2f}%"
        )

        print(
            f"📊 FAKE Confidence: "
            f"{fake_confidence:.2f}%"
        )

        print("\n========================================")

    except Exception as e:

        print(f"\n❌ Error: {e}")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    print("\n----------------------------------------")

    image_path = input(
        "\n📂 Enter Image Path (or type exit): "
    )

    if image_path.lower() == "exit":

        print("\n👋 Exiting Program...")

        break

    # ========================================================
    # CHECK FILE EXISTS
    # ========================================================

    if not os.path.exists(image_path):

        print("\n❌ File does not exist!")

        continue

    # ========================================================
    # RUN PREDICTION
    # ========================================================

    predict_image(image_path)