import streamlit as st
import torch
import torch.nn as nn

from torchvision import transforms, models
from transformers import BertTokenizer, BertModel

from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal Disaster Detection System",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #0E1117;
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: #CFCFCF;
    font-size: 18px;
    margin-bottom: 40px;
}

.card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #30363D;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.4);
}

.result-card {
    background: linear-gradient(135deg, #1E3C72, #2A5298);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    color: white;
}

.fake-card {
    background: linear-gradient(135deg, #8E0E00, #1F1C18);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    color: white;
}

.real-card {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    color: white;
}

.metric-box {
    background-color: #21262D;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LABELS
# ============================================================

classes = [
    'earthquake',
    'fire',
    'flood',
    'hurricane'
]

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased'
)

# ============================================================
# DISASTER MODEL
# ============================================================

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()

bert = BertModel.from_pretrained(
    'bert-base-uncased'
)

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

            nn.Linear(512, len(classes))
        )

    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):

        img_feat = self.image_model(image)

        text_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_feat = text_out.pooler_output

        combined = torch.cat(
            (img_feat, text_feat),
            dim=1
        )

        return self.fc(combined)

# ============================================================
# LOAD DISASTER MODEL
# ============================================================

@st.cache_resource
def load_disaster_model():

    model = MultimodalModel().to(device)

    checkpoint = torch.load(
        "checkpoints/best_epoch_9.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model.eval()

    return model

# ============================================================
# LOAD FAKE DETECTOR
# ============================================================

@st.cache_resource
def load_fake_detector():

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

    checkpoint = torch.load(
        "fake_checkpoints/best_fake_detector.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model

# ============================================================
# LOAD MODELS
# ============================================================

disaster_model = load_disaster_model()

fake_detector = load_fake_detector()

# ============================================================
# FAKE DETECTION FUNCTION
# ============================================================

def predict_fake_real(image):

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = fake_detector(image_tensor)

        probs = torch.softmax(outputs, dim=1)

    fake_confidence = probs[0][0].item() * 100

    real_confidence = probs[0][1].item() * 100

    if real_confidence >= 80:

        prediction = "REAL"

        confidence = real_confidence

    elif real_confidence >= 60:

        prediction = "SUSPICIOUS"

        confidence = real_confidence

    else:

        prediction = "FAKE"

        confidence = fake_confidence

    return prediction, confidence, real_confidence, fake_confidence

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 System Information")

    st.markdown("---")

    st.write("### 🤖 Models Used")

    st.write("""
    - EfficientNet-B0
    - ResNet50
    - BERT Base Uncased
    """)

    st.markdown("---")

    st.write("### 🎯 Features")

    st.write("""
    ✅ Fake Image Detection  
    ✅ Disaster Classification  
    ✅ Multimodal Learning  
    ✅ Real-Time Prediction  
    """)

    st.markdown("---")

    st.write("### ⚡ Device")

    st.success(device)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌍 Multimodal Disaster Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-Powered Fake Image Detection + Disaster Classification</div>',
    unsafe_allow_html=True
)

# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Disaster Image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    text_input = st.text_area(
        "📝 Enter Disaster Description",
        height=170
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MAIN PIPELINE
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ========================================================
    # FAKE DETECTION
    # ========================================================

    with st.spinner(
        "🔍 Analyzing image authenticity..."
    ):

        prediction, confidence, real_conf, fake_conf = (
            predict_fake_real(image)
        )

    st.markdown("## 🧠 Fake/Real Detection")

    if prediction == "REAL":

        st.markdown(f"""
        <div class="real-card">
        <h2>✅ REAL HUMAN-CAPTURED IMAGE</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == "SUSPICIOUS":

        st.warning(
            f"⚠️ POSSIBLY AI-GENERATED ({confidence:.2f}%)"
        )

    else:

        st.markdown(f"""
        <div class="fake-card">
        <h2>❌ FAKE AI-GENERATED IMAGE</h2>
        <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="metric-box">📊 REAL Confidence</div>',
            unsafe_allow_html=True
        )

        st.progress(real_conf / 100)

        st.write(f"{real_conf:.2f}%")

    with col2:

        st.markdown(
            '<div class="metric-box">📊 FAKE Confidence</div>',
            unsafe_allow_html=True
        )

        st.progress(fake_conf / 100)

        st.write(f"{fake_conf:.2f}%")

    # ========================================================
    # STOP IF FAKE
    # ========================================================

    if prediction == "FAKE":

        st.error(
            """
            🚫 Disaster classification stopped.
            
            Uploaded image appears AI-generated.
            """
        )

    else:

        if text_input:

            with st.spinner(
                "🌪️ Detecting disaster type..."
            ):

                image_tensor = transform(image)

                image_tensor = (
                    image_tensor.unsqueeze(0)
                    .to(device)
                )

                enc = tokenizer(

                    text_input,

                    padding='max_length',

                    truncation=True,

                    max_length=128,

                    return_tensors='pt'
                )

                input_ids = enc['input_ids'].to(device)

                attention_mask = enc[
                    'attention_mask'
                ].to(device)

                with torch.no_grad():

                    outputs = disaster_model(

                        image_tensor,

                        input_ids,

                        attention_mask
                    )

                    probs = torch.softmax(
                        outputs,
                        dim=1
                    )

                    _, pred = torch.max(
                        outputs,
                        1
                    )

                disaster = classes[pred.item()]

                disaster_confidence = (
                    probs[0][pred.item()].item()
                    * 100
                )

            # ====================================================
            # DISASTER EMOJIS
            # ====================================================

            disaster_emoji = {
                "earthquake": "🌍",
                "fire": "🔥",
                "flood": "🌊",
                "hurricane": "🌪️"
            }

            emoji = disaster_emoji.get(disaster, "⚠️")

            st.markdown("## 🌍 Disaster Detection")

            st.markdown(f"""
            <div class="result-card">
            <h2>{emoji} {disaster.upper()}</h2>
            <h3>Confidence: {disaster_confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(disaster_confidence / 100)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <p style='color:gray'>
    Developed using Streamlit, PyTorch, EfficientNet, ResNet50 and BERT
    </p>
    </center>
    """,
    unsafe_allow_html=True
)