import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os
import torch.nn.functional as F
import io

class_map = {0: "Salmon", 1: "Trout"}

# Load model
@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes
    model.load_state_dict(torch.load("deployment/model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("Fish Classifier: Salmon or Trout?")
st.markdown("Upload your own image **or** choose a sample image below:")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

sample_dir = "deployment/sample_images"
sample_filenames = ["salmon_1.jpg", "trout_1.jpg", "salmon_2.jpg", "trout_2.jpg"]

st.markdown("### Sample Images:")
selected_sample = None
cols = st.columns(len(sample_filenames))

# Show sample images with same size and buttons
for i, filename in enumerate(sample_filenames):
    img_path = os.path.join(sample_dir, filename)
    image = Image.open(img_path).convert("RGB").resize((150, 150))  # resize for uniform display
    cols[i].image(image, use_container_width=True)
    if cols[i].button("Choose", key=filename):
        selected_sample = img_path

if uploaded_file is None and selected_sample is not None:
    with open(selected_sample, "rb") as f:
        uploaded_file = io.BytesIO(f.read())

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        result = class_map[predicted_class]

    st.markdown(f"### Prediction: **{result}**")
    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")
