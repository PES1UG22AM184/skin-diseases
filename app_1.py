import streamlit as st
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("Project-MONAI/MONET")
    model = AutoModel.from_pretrained("Project-MONAI/MONET").to(device)
    model.eval()
    return model, processor, device

model, processor, device = load_model()

st.title("🧬 MONET Skin Enzyme & Disease Detector")
st.write("Upload a medical skin image to analyze using MONET (Project MONAI).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing with MONET..."):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    st.success("✅ Feature vector extracted!")
    st.write("Feature vector shape:", embeddings.shape)
    st.json({"embedding_preview": embeddings[0][:5].tolist()})

