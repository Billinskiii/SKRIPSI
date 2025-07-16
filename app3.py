import streamlit as st
import torch
from torchvision import transforms, models
from transformers import (
    ViTForImageClassification,
    BeitForImageClassification,
    DeiTForImageClassification
)
from PIL import Image
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import pandas as pd

# 🌟 CONFIG
st.set_page_config(page_title="🌍 Remote Sensing Classifier", page_icon="🛰️", layout="wide")

# 🌱 Init session state
if "page" not in st.session_state:
    st.session_state.page = "home"

# 🎨 Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f8f2e5; font-family: "Lilita One", sans-serif; color: black; }
    h1, h2, h3, .markdown-text-container, .css-10trblm, .stTabs [data-baseweb="tab"] { color: black !important; }
    .stMetric label { color: black; }
    section[data-testid="stSidebar"] {
        background-color: #685d2e !important;
    }
    .stFileUploader label, .stTabs [data-baseweb="tab"] {
        color: black !important;
        font-weight: bold;
    }
    div.stButton > button {
        background-color: #685d2e;
        color: white !important;
        font-weight: bold;
    }
            /* Ganti background multiselect box */
    div[data-baseweb="select"] > div {
        background-color: #e0e0e0 !important;
    }

    /* Ganti background tag selected (kotak merah jadi abu) */
    div[data-baseweb="tag"] {
        background-color: #9e9e9e !important;
    }
    /* Ganti warna tulisan di tag jadi putih */
    div[data-baseweb="tag"] span {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# 📋 Label kelas
class_names = [
    'Airport', 'Beach', 'Bridge', 'Desert', 'Forest', 'Industrial', 'Parking', 'Pond', 'Port',
    'PublicSpace', 'RailwayStation', 'Residential', 'Resort', 'River'
]

# 🧰 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_models():
    cnn = SimpleCNN(num_classes=len(class_names))
    cnn.load_state_dict(torch.load("../models/cnn_model.pt", map_location=device))
    cnn.to(device).eval()

    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
    resnet.load_state_dict(torch.load("../models/resnet_model.pt", map_location=device))
    resnet.to(device).eval()

    efficientnet = models.efficientnet_b0(weights=None)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, len(class_names))
    efficientnet.load_state_dict(torch.load("../models/efficientnet_model.pt", map_location=device))
    efficientnet.to(device).eval()

    vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=len(class_names), ignore_mismatched_sizes=True)
    vit.load_state_dict(torch.load("../models/vit_model.pt", map_location=device))
    vit.to(device).eval()

    vgg = models.vgg16(weights=None)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, len(class_names))
    vgg.load_state_dict(torch.load("../models/vgg16_model.pt", map_location=device))
    vgg.to(device).eval()

    densenet = models.densenet121(weights=None)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, len(class_names))
    densenet.load_state_dict(torch.load("../models/densenet_model.pt", map_location=device))
    densenet.to(device).eval()

    convnext = models.convnext_base(weights=None)
    convnext.classifier[2] = nn.Linear(convnext.classifier[2].in_features, len(class_names))
    convnext.load_state_dict(torch.load("../models/convnext_model.pt", map_location=device))
    convnext.to(device).eval()

    beit = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224", num_labels=len(class_names), ignore_mismatched_sizes=True)
    beit.load_state_dict(torch.load("../models/beit_model.pt", map_location=device))
    beit.to(device).eval()

    deit = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224", num_labels=len(class_names), ignore_mismatched_sizes=True)
    deit.load_state_dict(torch.load("../models/deit_model.pt", map_location=device))
    deit.to(device).eval()

    # YOLO classification models
    yolo8 = YOLO("D:/KULIAH/Sem 8/SKRIPSI/CODE/FINAL/blanchon/Aids/runs/classify/train8/weights/best.pt")
    yolo11 = YOLO("D:/KULIAH/Sem 8/SKRIPSI/CODE/FINAL/blanchon/Aids/runs/classify/train11/weights/best.pt")
    yolo12 = YOLO("D:/KULIAH/Sem 8/SKRIPSI/CODE/FINAL/blanchon/Aids/runs/classify/train12/weights/best.pt")

    return {
        "Simple CNN": cnn,
        "ResNet50": resnet,
        "EfficientNet-B0": efficientnet,
        "Vision Transformer": vit,
        "VGG16": vgg,
        "DenseNet121": densenet,
        "ConvNeXt": convnext,
        "BEiT": beit,
        "DeiT": deit,
        "YOLOv8": yolo8,
        "YOLOv11": yolo11,
        "YOLOv12": yolo12
    }

models_dict = load_models()

# 🧪 Predict function
def predict(model, image_tensor):
    if 'ultralytics' in str(type(model)).lower():
        img_np = transforms.ToPILImage()(image_tensor).convert("RGB")
        results = model(np.array(img_np), imgsz=224)
        probs = results[0].probs.data
        pred = torch.argmax(probs)
        conf = probs[pred]
        return class_names[pred.item()], conf.item()
    else:
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            if hasattr(output, 'logits'):
                output = output.logits
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        return class_names[pred.item()], conf.item()

# ================================
# 📦 PAGE RENDERING
# ================================

if st.session_state.page == "home":
    st.title("🛰️ Remote Sensing Image Classifier")
    st.markdown("""
    Selamat datang di aplikasi **Remote Sensing Image Classifier**!
    
    Aplikasi ini menggunakan berbagai model deep learning modern untuk mengklasifikasikan citra penginderaan jauh.
    
    Cari model terbaik untuk mendeteksi kebutuhanmu!
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background-color:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
            <h3>📖 Apa itu Remote Sensing?</h3>
            <p>
            Remote sensing (<strong>penginderaan jauh</strong>) adalah teknologi untuk memperoleh informasi tentang permukaan bumi melalui citra satelit atau drone, <strong>tanpa kontak langsung</strong>.<br><br>
            Citra ini bermanfaat untuk pemetaan, pemantauan lingkungan, perencanaan wilayah, dan lain-lain.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)  # sedikit spasi

        # Tombol di bawah card penjelasan
        if st.button("🚀 Masuk ke Aplikasi", key="go_app"):
            st.session_state.page = "app"

    with col2:
        st.markdown("""
        <div style="background-color:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
            <h3>🏷️ 14 Class yang dapat dideteksi:</h3>
            <ul style="margin:0; padding-left:20px;">
                <li>Airport ✈️</li>
                <li>Beach 🏖️</li>
                <li>Bridge 🌉</li>
                <li>Desert 🏜️</li>
                <li>Forest 🌳</li>
                <li>Industrial 🏭</li>
                <li>Parking 🅿️</li>
                <li>Pond 🪷</li>
                <li>Port ⚓</li>
                <li>PublicSpace 🏙️</li>
                <li>RailwayStation 🚉</li>
                <li>Residential 🏘️</li>
                <li>Resort 🏝️</li>
                <li>River 🌊</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

elif st.session_state.page == "app":
    with st.sidebar:
        if st.button("← Home", key="go_home"):
            st.session_state.page = "home"
        
        # ⭐ Tambahkan tombol ke halaman About Us
        if st.button("Tentang Saya"):
            st.session_state.page = "about"
        
        st.header("Pemilihan Model")
        st.subheader("Models:")
        selected_models = []
        for model_name in models_dict.keys():
            if st.checkbox(model_name, value=True):
                selected_models.append(model_name)
        st.markdown("---")
        st.caption("")

    st.title("🛰️ Remote Sensing Image Classifier")
    st.caption("""
        ### 🧭 Panduan Penggunaan :
        1️⃣ **Pilih** model yang ingin dijalankan di sidebar kiri (default-nya semua model aktif).  
        2️⃣ **Unggah** citra JPG/PNG pada form di bawah.  
        3️⃣ **Tunggu** proses prediksi – lihat hasil prediksi & confidence dari setiap model di tab “📊 Predictions”.  
        4️⃣ **Bandingkan** confidence semua model di tab “📈 Chart”.
    
        💡 Kamu juga bisa lihat class dengan confidence tertinggi dan menganalisis performa setiap model!
    """)

    uploaded_file = st.file_uploader("Unggah Gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Gambar Terunggah", width=250)
        image_tensor = transform(image)

        tab1, tab2 = st.tabs(["📊 Predictions", "📈 Chart"])

        results = []
        with tab1:
            st.subheader("📊 Predictions:")
            with st.spinner("Predicting..."):
                cols = st.columns(2)
                temp_results = []
                for model_name in selected_models:
                    model = models_dict[model_name]
                    label, confidence = predict(model, image_tensor)
                    temp_results.append({"Model": model_name, "Label": label, "Confidence": confidence})

                best_model = max(temp_results, key=lambda x: x["Confidence"]) if temp_results else None

                for i, res in enumerate(temp_results):
                    is_best = (res["Model"] == best_model["Model"])
                    bg_color = "#d4edda" if is_best else "#ffffff"
                    badge_html = "<span style='color:white; background-color:#28a745; padding:2px 6px; border-radius:4px; font-size:12px; margin-left:8px;'>🔥 Best Model</span>" if is_best else ""

                    with cols[i % 2]:
                        st.markdown(f"""
                        <div style="
                            background-color: {bg_color};
                            padding: 15px;
                            margin-bottom: 10px;
                            border-radius: 10px;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        ">
                            <h4 style='margin-bottom:8px;'>{res["Model"]} {badge_html}</h4>
                            <p style='margin:0;'>🏷️ <strong>{res["Label"]}</strong></p>
                            <p style='margin:0;'>🔍 Confidence: <strong>{res["Confidence"]:.2%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                results = temp_results



        with tab2:
            if results:
                import plotly.graph_objects as go

                df = pd.DataFrame(results).sort_values(by="Confidence", ascending=False)

                # Tentukan best model
                best_model_name = df.iloc[0]["Model"]

                # Tambahkan kolom warna: hijau jika best model, else abu
                df["Color"] = df["Model"].apply(
                    lambda x: "green" if x == best_model_name else "#a3a3a3"
                )

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=df["Model"],
                            y=df["Confidence"],
                            marker_color=df["Color"]
                        )
                    ]
                )

                # Layout: background putih & style clean
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    yaxis=dict(
                        title=dict(text='Confidence', font=dict(color='black')),
                        tickformat=".0%",
                        showgrid=True,
                        gridcolor='#000000',
                        tickfont=dict(color='black')
                    ),
                    xaxis=dict(
                        title=dict(text='Model', font=dict(color='black')),
                        tickfont=dict(color='black')
                    ),
                    title=dict(text='Confidence per Model', font=dict(color='black'))
                )

                st.plotly_chart(fig, use_container_width=True)
elif st.session_state.page == "about":
    st.title("👨‍💻 Tentang Saya")
    
    # Biodata
    st.markdown("""
    <div style="background-color:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
        <h3>🙋‍♂️ Farhan Rizky Novan Dinaputra</h3>
        <ul style="list-style-type: none; padding-left:0;">
            <li><strong>📍 Asal:</strong> Bandung, Indonesia</li>
            <li><strong>🎓 Mahasiswa:</strong> S1 Teknik Informatika, Binus University</li>
            <li><strong>📧 Email:</strong> farhan.dinaputra@gmail.com</li>
            <li><strong>💡 Minat:</strong> Deep Learning, Computer Vision, Data Engineering</li>
        </ul>
        <p>Saya sedang mengerjakan skripsi mengenai analisis performansi berbagai model deep learning untuk klasifikasi citra penginderaan jauh. 
        Aplikasi ini adalah hasil dari riset tersebut untuk membandingkan performa antar model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tombol kembali ke halaman sebelumnya
    if st.button("← Kembali ke Aplikasi", key="back_to_app"):
        st.session_state.page = "app"