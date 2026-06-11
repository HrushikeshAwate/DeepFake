import streamlit as st
import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Force Matplotlib to use a headless/non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# XAI Packages
import shap
from captum.attr import LayerGradCam
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Set clean page configuration
st.set_page_config(
    page_title="Multi-Class Deepfake Forensic Suite", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Multi-Class Deepfake Forensic Verification Dashboard")
st.markdown("Upload a facial profile image to compute cross-framework feature attributions (Grad-CAM, LIME, SHAP) and visualize structural anomalies.")

# Global configurations
class_names = ['Real', 'StyleGAN', 'Diffusion', 'FaceSwap']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation profiles
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Model Architecture (Cached to prevent reload delays on user actions)
@st.cache_resource
def load_forensic_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    
    # Searches directly in the root directory of your GitHub repository deployment
    weights_path = 'best_deepfake_model.pth'
        
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        st.sidebar.error(f"⚠️ Warning: '{weights_path}' not found in the root directory. Please upload it to your GitHub repository.")
        
    model = model.to(device)
    model.eval()
    return model

try:
    model = load_forensic_model()
    st.sidebar.success("✅ Forensic Network Weights Active!")
except Exception as e:
    st.sidebar.error(f"❌ Initialization Error: {e}")

# Map normalizer utility
def normalize_map(attr_map):
    attr_min, attr_max = attr_map.min(), attr_map.max()
    if attr_max - attr_min > 1e-5:
        return (attr_map - attr_min) / (attr_max - attr_min)
    return np.zeros_like(attr_map)

# Dashboard File Uploader
uploaded_file = st.file_uploader("📂 Select an input facial image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Process and render raw uploads
    raw_image = Image.open(uploaded_file).convert("RGB")
    transformed_tensor = val_test_transform(raw_image)
    img_tensor = transformed_tensor.unsqueeze(0).to(device)
    original_img_np = np.array(raw_image.resize((224, 224))) / 255.0

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(raw_image, caption="Uploaded Target Profile", use_container_width=True)
        
    # 2. Run Network Forward Pass Prediction
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_class = np.argmax(probs)

    with col2:
        st.subheader("📊 Classification Analysis Verdict")
        for i, name in enumerate(class_names):
            if i == pred_class:
                st.markdown(f"**🔹 {name}: {probs[i]*100:.2f}% (Predicted Category)**")
            else:
                st.markdown(f"🔸 {name}: {probs[i]*100:.2f}%")
        st.progress(float(probs[pred_class]))

    # 3. Compute Multi-Framework Interpretability Arrays
    st.markdown("---")
    st.subheader("⚡ Computing Cross-Framework Ensemble Attributions...")
    
    with st.spinner("Processing Grad-CAM, LIME, and GradientSHAP layers..."):
        # A. Grad-CAM Calculation
        target_layer = model.features[-1]
        lgc = LayerGradCam(model, target_layer)
        gc_attr = lgc.attribute(img_tensor, target=int(pred_class))
        gc_upsampled = LayerGradCam.interpolate(gc_attr, (224, 224), interpolate_mode="bilinear")
        gc_map = gc_upsampled.squeeze().cpu().detach().numpy()
        if len(gc_map.shape) == 3:
            gc_map = np.mean(gc_map, axis=0)
        gc_map_norm = normalize_map(np.abs(gc_map))

        # B. LIME continuous conversion
        lime_explainer = lime_image.LimeImageExplainer()
        def batch_predict(images):
            batch = torch.stack([val_test_transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images], dim=0).to(device)
            with torch.no_grad():
                outputs = model(batch)
                predictions = torch.nn.functional.softmax(outputs, dim=1)
            return predictions.cpu().numpy()
            
        explanation = lime_explainer.explain_instance(original_img_np, batch_predict, top_labels=4, hide_color=0, num_samples=100)
        lime_map = np.zeros((224, 224))
        segments = explanation.segments
        for seg_id, weight in explanation.local_exp[int(pred_class)]:
            lime_map[segments == seg_id] = np.abs(weight)
        lime_map_norm = normalize_map(lime_map)

        # C. GradientSHAP Calculation
        from captum.attr import GradientShap
        shap_engine = GradientShap(model)
        background = torch.zeros(1, 3, 224, 224).to(device)
        shap_attr = shap_engine.attribute(img_tensor, baselines=background, target=int(pred_class))
        shap_map = np.mean(np.abs(shap_attr.squeeze().cpu().detach().numpy()), axis=0)
        shap_map_norm = normalize_map(shap_map)

        # D. Ensemble Mathematical Combination
        ensemble_average_map = (gc_map_norm + lime_map_norm + shap_map_norm) / 3.0

    # 4. Construct Graphical Plots Layout Display
    st.subheader("🖼️ Comparative XAI Heatmap Matrix Panels")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 1: Grad-CAM
    axes[0].imshow(original_img_np)
    axes[0].imshow(gc_map_norm, cmap='jet', alpha=0.5)
    axes[0].set_title("Grad-CAM Thermal Focus")
    axes[0].axis('off')

    # Panel 2: LIME
    axes[1].imshow(original_img_np)
    axes[1].imshow(lime_map_norm, cmap='jet', alpha=0.5)
    axes[1].set_title("LIME Feature Segments")
    axes[1].axis('off')

    # Panel 3: SHAP
    axes[2].imshow(original_img_np)
    axes[2].imshow(shap_map_norm, cmap='jet', alpha=0.5)
    axes[2].set_title("SHAP Pixel Densities")
    axes[2].axis('off')

    # Panel 4: Unified Ensemble Average Blend
    axes[3].imshow(original_img_np)
    axes[3].imshow(ensemble_average_map, cmap='hot', alpha=0.6)
    axes[3].set_title("📊 ENSEMBLE AVERAGE BLEND")
    axes[3].axis('off')

    # Render layout safely inside Streamlit
    st.pyplot(fig)
    plt.close(fig) # Clear memory references
