# ================== SAME IMPORTS ==================
import torch
import pandas as pd
import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ================== DATA LOADING ==================
DATA_PATH = "E:\\PAAIProject\\CrisisMMD_v2.0\\CrisisMMD_v2.0\\"
ANNOTATION_PATH = DATA_PATH + "annotations/"

files = [f for f in os.listdir(ANNOTATION_PATH) if not f.startswith("._")]

df_list = []
for file in files:
    temp_df = pd.read_csv(ANNOTATION_PATH + file, sep="\t")
    disaster_type = file.split("_final")[0]
    temp_df["disaster_type"] = disaster_type
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# ================== CLEANING ==================
df = df.rename(columns={"image_id": "image"})
df = df[['image', 'tweet_text', 'disaster_type']]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

df['tweet_text'] = df['tweet_text'].apply(clean_text)
df = df.dropna()
df = df[df['tweet_text'].str.strip() != ""]

# ================== 🔥 LABEL SIMPLIFICATION (KEY STEP) ==================
def simplify_label(label):
    if "hurricane" in label:
        return "hurricane"
    elif "earthquake" in label:
        return "earthquake"
    elif "flood" in label:
        return "flood"
    elif "fire" in label:
        return "fire"
    else:
        return "other"

df['disaster_type'] = df['disaster_type'].apply(simplify_label)

# ❌ Remove less useful class
df = df[df['disaster_type'] != "other"]

# ================== LABEL ENCODING ==================
le = LabelEncoder()
df['label'] = le.fit_transform(df['disaster_type'])

print("Classes:", le.classes_)   # 🔥 for confirmation

# ================== SPLIT ==================
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)

weights = torch.tensor(weights, dtype=torch.float)

# ================== TRANSFORM ==================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import glob

class CrisisDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row['image'] + ".jpg"

#  FIXED SEARCH (works with grouped labels)
        search_path = DATA_PATH + f"data_image/**/{img_name}"
        matches = glob.glob(search_path, recursive=True)

#  SAFE fallback (no infinite loop)
        if len(matches) == 0:
            import random
            return self.__getitem__(random.randint(0, len(self.df)-1))

        image = Image.open(matches[0]).convert('RGB')
        image = transform(image)

        text = row['tweet_text']
        enc = tokenizer(text, padding='max_length',
                        truncation=True, max_length=128,
                        return_tensors='pt')

        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()

        label = row['label']

        return image, input_ids, attention_mask, label

# ================== LOADERS ==================
train_loader = DataLoader(CrisisDataset(train_df), batch_size=16, shuffle=True, drop_last=True)

# ================== MODEL ==================
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()

bert = BertModel.from_pretrained('bert-base-uncased')

#  PARTIAL UNFREEZE
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.layer4.parameters():
    param.requires_grad = True

for param in bert.parameters():
    param.requires_grad = False
for param in bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = resnet
        self.text_model = bert
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(le.classes_))
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_model(image)

        text_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_out.pooler_output

        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fc(combined)

# ================== DEVICE ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = MultimodalModel().to(device)

# ================== OPTIMIZER ==================
optimizer = torch.optim.AdamW([
    {'params': model.image_model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.text_model.encoder.layer[-2:].parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])

criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# ================== TRAIN ==================
EPOCHS = 10   

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for img, input_ids, mask, labels in tqdm(train_loader):

        img = img.to(device)
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        outputs = model(img, input_ids, mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    torch.save({
        'model_state_dict': model.state_dict()
    }, f"checkpoints/best_epoch_{epoch}.pth")