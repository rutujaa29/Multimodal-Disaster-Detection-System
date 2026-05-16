import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

CHECKPOINT_DIR = "fake_checkpoints"
PREPARED_DIR = "prepared_dataset"

BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\n🚀 Using Device: {DEVICE}")

# ============================================================
# STEP 1: PREPARE DATASET
# ============================================================

def prepare_dataset(input_dir, output_dir):
    """
    Keeps REAL images unchanged
    Upscales ONLY FAKE images to 512x512
    """

    if os.path.exists(output_dir):

        print(f"\n✅ Prepared dataset already exists: {output_dir}")

        return

    print("\n🔥 Preparing Dataset...\n")

    for split in ["train", "val", "test"]:

        for cls in ["real", "fake"]:

            input_class_dir = os.path.join(
                input_dir,
                split,
                cls
            )

            output_class_dir = os.path.join(
                output_dir,
                split,
                cls
            )

            os.makedirs(output_class_dir, exist_ok=True)

            image_files = os.listdir(input_class_dir)

            for img_name in tqdm(
                image_files,
                desc=f"{split}/{cls}"
            ):

                input_path = os.path.join(
                    input_class_dir,
                    img_name
                )

                output_path = os.path.join(
                    output_class_dir,
                    img_name
                )

                try:

                    img = Image.open(input_path).convert("RGB")

                    # ====================================================
                    # UPSCALE ONLY FAKE IMAGES
                    # ====================================================

                    if cls == "fake":

                        img = img.resize(
                            (512, 512),
                            Image.LANCZOS
                        )

                    # REAL IMAGES REMAIN ORIGINAL
                    img.save(output_path)

                except Exception as e:

                    print(f"❌ Error processing {img_name}: {e}")

    print("\n✅ Dataset Preparation Completed!")

# ============================================================
# RUN DATASET PREPARATION
# ============================================================

prepare_dataset(DATASET_PATH, PREPARED_DIR)

# ============================================================
# STEP 2: DATA AUGMENTATION
# ============================================================

train_transforms = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.RandomPerspective(
        distortion_scale=0.15,
        p=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# STEP 3: LOAD DATASETS
# ============================================================

print("\n📂 Loading Dataset...\n")

train_dataset = datasets.ImageFolder(
    os.path.join(PREPARED_DIR, "train"),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(PREPARED_DIR, "val"),
    transform=val_test_transforms
)

test_dataset = datasets.ImageFolder(
    os.path.join(PREPARED_DIR, "test"),
    transform=val_test_transforms
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"✅ Train Images: {len(train_dataset)}")
print(f"✅ Validation Images: {len(val_dataset)}")
print(f"✅ Test Images: {len(test_dataset)}")

print(f"\n📌 Classes: {train_dataset.classes}")

# ============================================================
# STEP 4: LOAD MODEL
# ============================================================

print("\n🧠 Loading EfficientNet-B0...\n")

model = models.efficientnet_b0(weights="DEFAULT")

# ============================================================
# FREEZE EARLY LAYERS
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False

# ============================================================
# UNFREEZE LAST BLOCKS
# ============================================================

for param in model.features[-3:].parameters():
    param.requires_grad = True

# ============================================================
# CUSTOM CLASSIFIER
# ============================================================

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(

    nn.Dropout(0.4),

    nn.Linear(num_features, 256),

    nn.ReLU(),

    nn.BatchNorm1d(256),

    nn.Dropout(0.3),

    nn.Linear(256, 2)
)

model = model.to(DEVICE)

print("\n✅ Model Ready!")

# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

# ============================================================
# EARLY STOPPING
# ============================================================

early_stop_counter = 0
PATIENCE = 4

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    loop = tqdm(loader)

    for images, labels in loop:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    accuracy = 100 * correct / total

    return running_loss / len(loader), accuracy

# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return running_loss / len(loader), accuracy

# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0

print("\n🚀 STARTING TRAINING...\n")

for epoch in range(NUM_EPOCHS):

    print(f"\n📌 Epoch [{epoch+1}/{NUM_EPOCHS}]")

    # ========================================================
    # TRAINING
    # ========================================================

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_loss, val_acc = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_acc)

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(f"\n🔥 Train Loss: {train_loss:.4f}")

    print(f"✅ Train Accuracy: {train_acc:.2f}%")

    print(f"\n📊 Validation Loss: {val_loss:.4f}")

    print(f"🎯 Validation Accuracy: {val_acc:.2f}%")

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_acc > best_val_accuracy:

        best_val_accuracy = val_acc

        early_stop_counter = 0

        torch.save({

            "model_state_dict": model.state_dict(),

            "val_accuracy": val_acc,

            "epoch": epoch + 1

        }, os.path.join(
            CHECKPOINT_DIR,
            "best_fake_detector.pth"
        ))

        print("\n💾 BEST MODEL SAVED!")

    else:

        early_stop_counter += 1

        print(
            f"\n⏳ Early Stop Counter: "
            f"{early_stop_counter}/{PATIENCE}"
        )

        if early_stop_counter >= PATIENCE:

            print("\n🛑 EARLY STOPPING TRIGGERED!")

            break

# ============================================================
# FINAL TESTING
# ============================================================

print("\n🧪 FINAL TESTING...\n")

checkpoint = torch.load(
    os.path.join(
        CHECKPOINT_DIR,
        "best_fake_detector.pth"
    )
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_acc = evaluate(
    model,
    test_loader
)

print(f"\n🎯 FINAL TEST ACCURACY: {test_acc:.2f}%")

print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")