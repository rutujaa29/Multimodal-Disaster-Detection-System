import torch
import pandas as pd
import os
import re
import glob
from PIL import Image
import torch.nn as nn

from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------ PATH ------------------
DATA_PATH = "E:\\PAAIProject\\CrisisMMD_v2.0\\CrisisMMD_v2.0\\"

# ------------------ TEXT CLEANING ------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# ------------------ LABEL SIMPLIFICATION ------------------
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

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ TOKENIZER ------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ------------------ LOAD DATA ------------------
ANNOTATION_PATH = DATA_PATH + "annotations/"
files = [f for f in os.listdir(ANNOTATION_PATH) if not f.startswith("._")]

df_list = []
for file in files:
    temp_df = pd.read_csv(ANNOTATION_PATH + file, sep="\t")
    disaster_type = file.split("_final")[0]
    temp_df["disaster_type"] = disaster_type
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

df = df.rename(columns={"image_id": "image"})
df = df[['image', 'tweet_text', 'disaster_type']]

df['tweet_text'] = df['tweet_text'].apply(clean_text)
df = df.dropna()
df = df[df['tweet_text'].str.strip() != ""]

# 🔥 SAME LABEL PROCESSING AS TRAINING
df['disaster_type'] = df['disaster_type'].apply(simplify_label)
df = df[df['disaster_type'] != "other"]

# ------------------ LABEL ENCODING ------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['disaster_type'])

print("Classes:", le.classes_)

# ------------------ SPLIT ------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

# ------------------ DATASET ------------------
class CrisisDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row['image'] + ".jpg"

        search_path = DATA_PATH + f"data_image/**/{img_name}"
        matches = glob.glob(search_path, recursive=True)

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

# ------------------ LOADER ------------------
test_loader = DataLoader(CrisisDataset(test_df), batch_size=16, shuffle=False)

# ------------------ MODEL ------------------
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()

bert = BertModel.from_pretrained('bert-base-uncased')

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

# ------------------ LOAD MODEL ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalModel().to(device)

# 🔥 LOAD LAST/BEST CHECKPOINT
checkpoint = torch.load("checkpoints/best_epoch_9.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print("✅ Model Loaded Successfully")

# ------------------ EVALUATION ------------------
correct = 0
total = 0

with torch.no_grad():
    for images, input_ids, mask, labels in test_loader:

        images = images.to(device)
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        outputs = model(images, input_ids, mask)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n🔥 Test Accuracy: {accuracy:.2f}%")

# ------------------ SAMPLE OUTPUT ------------------
print("\n🔍 Sample Predictions:\n")

with torch.no_grad():
    for images, input_ids, mask, labels in test_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        mask = mask.to(device)

        outputs = model(images, input_ids, mask)
        _, predicted = torch.max(outputs, 1)

        for i in range(5):
            actual = le.inverse_transform([labels[i].item()])[0]
            pred = le.inverse_transform([predicted[i].item()])[0]

            print(f"Actual: {actual}  |  Predicted: {pred}")

        break