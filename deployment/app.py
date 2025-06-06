import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os
import torch.nn.functional as F
import random

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

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("Fish Classifier: Salmon or Trout?")
st.markdown("Upload your own image **or** select one of our sample images below.")

sample_dir = os.path.join("deployment", "sample_images")
sample_filenames = ["salmon_1.jpg", "salmon_2.jpg", "trout_1.jpg", "trout_2.jpg"]
random.shuffle(sample_filenames)

selected_sample = st.selectbox(
    "Choose a sample image (optional):",
    ["None"] + sample_filenames,
    format_func=lambda x: x if x == "None" else f"Sample: {x}"
)

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

image = None

if uploaded_file is None and selected_sample != "None":
    image_path = os.path.join(sample_dir, selected_sample)
    image = Image.open(image_path).convert("RGB")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

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
