import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os
import torch.nn.functional as F 


# Class map
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

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Fish Classifier: Salmon or Trout?")
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        result = class_map[predicted_class]

    st.markdown(f"### Prediction: **{result}** ({predicted_class})")
    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")
