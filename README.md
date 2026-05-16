# 🌍 Multimodal Disaster Detection System

An AI-powered multimodal disaster analysis system that combines:

* 🧠 Fake vs Real Image Detection
* 🌪️ Disaster Type Classification
* 🖼️ Image Understanding
* 📝 Text Understanding
* ⚡ Real-Time Streamlit Dashboard

This project uses Deep Learning, Computer Vision, Natural Language Processing (NLP), and Multimodal Learning techniques to analyze disaster-related content from both images and text.

---

# 📌 Project Overview

Social media platforms often contain:

* Real disaster images
* AI-generated fake disaster images
* Misleading information
* Non-informative disaster posts

This project aims to solve this problem by building a two-stage AI pipeline:

## ✅ Stage 1 — Fake vs Real Image Detection

The uploaded image is first analyzed using an EfficientNet-B0 based deep learning model.

The system determines whether the uploaded image is:

* ✅ Real Human-Captured Image
* ❌ AI-Generated Fake Image
* ⚠️ Suspicious / Possibly AI-Generated

If the image is detected as fake, the pipeline stops further disaster analysis.

---

## ✅ Stage 2 — Disaster Detection Using Multimodal Learning

If the uploaded image is real, the system performs multimodal disaster classification using:

* 🖼️ Image Features from ResNet50
* 📝 Text Features from BERT

The model combines both image and text information to predict the disaster type.

Supported disaster classes:

* 🌊 Flood
* 🔥 Fire
* 🌍 Earthquake
* 🌪️ Hurricane

---

# 🧠 Technologies Used

| Technology      | Purpose                   |
| --------------- | ------------------------- |
| Python          | Core Programming Language |
| PyTorch         | Deep Learning Framework   |
| Streamlit       | Web Application UI        |
| EfficientNet-B0 | Fake Image Detection      |
| ResNet50        | Image Feature Extraction  |
| BERT            | Text Feature Extraction   |
| Transformers    | NLP Processing            |
| PIL             | Image Processing          |
| Torchvision     | Computer Vision Utilities |

---

# 🏗️ System Architecture

```text
Uploaded Image + Text
            ↓
Stage 1: Fake vs Real Detection
            ↓
If REAL
            ↓
Stage 2: Multimodal Disaster Classification
            ↓
Final Disaster Prediction
```

---

# 📂 Project Structure

```text
PAAIProject/
│
├── app.py
├── train_model.py
├── train_fake_model.py
├── test_model.py
├── test_fake_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── checkpoints/
│   └── best_epoch_9.pth
│
├── fake_checkpoints/
│   └── best_fake_detector.pth
│
├── dataset/
├── prepared_dataset/
└── venv/
```

---

# 📥 Download Trained Model Weights

The trained model files are too large to upload directly to GitHub.

Download the trained model weights from Google Drive:

## 🔗 Google Drive Link

https://drive.google.com/drive/folders/1LEI0BoJZEa6jwNBS-NyT_9Hy9mG3K-D5?usp=sharing

---

# 📌 After Downloading

Place the downloaded files in the following folders:

## 1️⃣ Disaster Detection Model

Place:

```text
best_epoch_9.pth
```

inside:

```text
checkpoints/
```

---

## 2️⃣ Fake Image Detection Model

Place:

```text
best_fake_detector.pth
```

inside:

```text
fake_checkpoints/
```

---

# ⚙️ Installation Steps

## Step 1 — Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
```

---

## Step 2 — Open Project Folder

```bash
cd PAAIProject
```

---

## Step 3 — Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 4 — Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

After running the command, Streamlit will generate a local URL such as:

```text
http://localhost:8501
```

Open the link in your browser.

---

# 🧪 Testing the System

## Example Flood Description

```text
Severe flooding has submerged roads and houses after continuous heavy rainfall. Rescue teams are evacuating stranded people using boats.
```

---

## Example Fire Description

```text
Massive wildfire spreading rapidly through dry forest areas with thick smoke covering the sky and firefighters trying to control the flames.
```

---

## Example Earthquake Description

```text
Several buildings collapsed after a strong earthquake hit the city. Rescue teams are searching for trapped civilians.
```

---

## Example Hurricane Description

```text
Powerful storm winds and heavy rainfall caused severe destruction across coastal regions with flooding and damaged infrastructure.
```

---

# 📊 Model Details

## 🔹 Fake Image Detection Model

| Component         | Details                |
| ----------------- | ---------------------- |
| Architecture      | EfficientNet-B0        |
| Task              | Fake vs Real Detection |
| Framework         | PyTorch                |
| Training Accuracy | ~95%                   |
| Test Accuracy     | ~95%                   |

---

## 🔹 Disaster Detection Model

| Component      | Details                            |
| -------------- | ---------------------------------- |
| Image Backbone | ResNet50                           |
| Text Backbone  | BERT Base Uncased                  |
| Fusion Method  | Feature Concatenation              |
| Task           | Disaster Classification            |
| Classes        | Flood, Fire, Earthquake, Hurricane |

---

# 🔍 How Multimodal Fusion Works

The project combines:

* Image features extracted by ResNet50
* Text features extracted by BERT

These features are concatenated together and passed through fully connected neural network layers for final disaster prediction.

This allows the model to understand:

* Visual disaster patterns
* Textual disaster descriptions

simultaneously.

---

# 🎯 Features of the System

✅ Fake vs Real Image Detection

✅ AI-generated Disaster Image Detection

✅ Multimodal Learning

✅ Real-Time Predictions

✅ Modern Streamlit Dashboard

✅ Deep Learning Based Architecture

✅ Transfer Learning

✅ Image + Text Fusion

---

# 🚀 Future Improvements

* Add more disaster categories
* Improve fake image generalization
* Add video disaster analysis
* Deploy on cloud platforms
* Add real-time social media monitoring
* Add heatmap visualizations

---

# 📸 Application Workflow

1️⃣ User uploads disaster image

2️⃣ User enters disaster description

3️⃣ System checks if image is fake or real

4️⃣ If real → disaster classification starts

5️⃣ Final disaster prediction is displayed

---

# 👨‍💻 Developed Using

* PyTorch
* Streamlit
* Transformers
* EfficientNet
* ResNet50
* BERT
* Python

---

# 📌 Note

This project is developed for educational and research purposes related to:

* Disaster Management
* Misinformation Detection
* Artificial Intelligence
* Multimodal Deep Learning

---

# ⭐ If You Like This Project

Please consider giving this repository a star ⭐
