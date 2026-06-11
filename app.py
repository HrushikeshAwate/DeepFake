import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import shap
from PIL import Image
from lime import lime_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =====================================================================
# 1. PAGE SETUP & CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Multi-Class Deepfake Analyzer", layout="wide")
st.title("🛡️ Multi-Class Deepfake Detection & XAI Hub")
st.write("Upload a facial image to determine if it is **Real**, a **FaceSwap**, or **Diffusion-Generated**, and inspect the XAI structural reasoning.")

# =====================================================================
# 2. MODEL LOADING WITH CLOUD WEIGHTS AUTOMATION
# =====================================================================
@st.cache_resource
def load_deepfake_detector():
    # Rebuild your 3-class model architecture (EfficientNet Base)
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3) # 3 outputs: Real, FaceSwap, Diffusion
    
    # Weights configuration file hook
    weights_path = "model_weights.pth"
    
    # 💡 AUTOMATED DOWNLOAD HOOK: If you end up hosting your weights file on Google Drive 
    # because it is over 100MB, uncomment the lines below and add 'gdown' to your requirements.txt:
    # import gdown
    # if not os.path.exists(weights_path):
    #     with st.spinner("Downloading 3-Class Model Weights from Cloud Storage..."):
    #         file_id = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
    #         gdown.download(f'https://drive.google.com/uc?id={file_id}', weights_path, quiet=False)

    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            st.sidebar.success("✅ 3-Class Model Weights Loaded Successfully!")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading weights file matrix: {e}")
    else:
        st.sidebar.warning("⚠️ Running on randomized weights. Place your 'model_weights.pth' file in your GitHub repository.")

    model.eval()
    
    # Standard evaluation pre-processing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize XAI backend modules
    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    lime_explainer = lime_image.LimeImageExplainer()
    
    def shap_predict(images):
        tensor_input = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        with torch.no_grad():
            logits = model(tensor_input)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.numpy()
        
    shap_explainer = shap.Explainer(shap_predict, shap.maskers.Image("inpaint_telea", (224, 224, 3)))
    
    return model, transform, cam, lime_explainer, shap_explainer

model, transform, cam, lime_explainer, shap_explainer = load_deepfake_detector()

# =====================================================================
# 3. HEATMAP STANDARDIZATION UTILITIES
# =====================================================================
def standardize_heatmap(raw_mask):
    if raw_mask is None: return np.zeros((224, 224), dtype=np.uint8)
    if len(raw_mask.shape) == 3: raw_mask = np.mean(raw_mask, axis=-1)
    resized = cv2.resize(raw_mask, (224, 224), interpolation=cv2.INTER_LINEAR)
    min_val, max_val = np.min(resized), np.max(resized)
    normalized = (resized - min_val) / (max_val - min_val) if max_val - min_val > 0 else np.zeros((224, 224))
    return np.uint8(255 * normalized)

def lime_predict_bridge(images):
    batch = torch.stack([transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images], dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.numpy()

# =====================================================================
# 4. USER INTERFACE & METRICS INTERACTION
# =====================================================================
uploaded_file = st.file_uploader("Drop a facial file image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Target Forensic File", use_container_width=True)
        
    # Tensor formatting maps
    img_tensor = transform(image).unsqueeze(0)
    img_np = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0

    # Model inference
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0].numpy()
        
    class_labels = ["Real Face", "FaceSwap Deepfake", "Diffusion-Generated Fake"]
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx] * 100

    with col2:
        st.subheader("Classification Matrix Output")
        st.metric(label="Primary Classification Verdict", value=class_labels[predicted_class_idx], delta=f"{confidence:.2f}% System Confidence")
        
        st.write("**Full Class Probability Distribution:**")
        st.progress(float(probabilities[0]), text=f"🟢 Real: {probabilities[0]*100:.1f}%")
        st.progress(float(probabilities[1]), text=f"🔴 FaceSwap: {probabilities[1]*100:.1f}%")
        st.progress(float(probabilities[2]), text=f"🟡 Diffusion: {probabilities[2]*100:.1f}%")

    # =====================================================================
    # 5. EXPLAINABLE AI HEATMAP GENERATION
    # =====================================================================
    st.markdown("---")
    st.subheader("💡 Explainable AI (XAI) Feature Attribution Grid")
    st.write("Evaluating localized artifacts causing prediction logit weights...")
    
    with st.spinner("Processing background mathematical perturbations..."):
        # 1. Grad-CAM Calculation
        cam_raw = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(predicted_class_idx)])[0]
        gradcam_map = standardize_heatmap(cam_raw)
        gradcam_colored = cv2.applyColorMap(gradcam_map, cv2.COLORMAP_JET)
        gradcam_colored = cv2.cvtColor(gradcam_colored, cv2.COLOR_BGR2RGB)
        
        # 2. LIME Superpixel Calculation (Fast configuration for quick web response)
        exp = lime_explainer.explain_instance(img_np, lime_predict_bridge, top_labels=1, num_samples=35)
        _, lime_mask = exp.get_image_and_mask(label=predicted_class_idx, positive_only=True, hide_rest=False)
        lime_map = standardize_heatmap(lime_mask)
        lime_colored = cv2.applyColorMap(lime_map, cv2.COLORMAP_JET)
        lime_colored = cv2.cvtColor(lime_colored, cv2.COLOR_BGR2RGB)
        
        # 3. SHAP Game-Theoretic Value Calculation
        shap_values = shap_explainer(img_np[np.newaxis, ...], max_evals=35)
        shap_raw = np.abs(shap_values.values[0, :, :, :, predicted_class_idx]).mean(axis=-1)
        shap_map = standardize_heatmap(shap_raw)
        shap_colored = cv2.applyColorMap(shap_map, cv2.COLORMAP_JET)
        shap_colored = cv2.cvtColor(shap_colored, cv2.COLOR_BGR2RGB)

    # Display XAI maps in a horizontal layout matrix
    xai_col1, xai_col2, xai_col3 = st.columns(3)
    with xai_col1:
        st.image(gradcam_colored, caption="Grad-CAM (CNN Feature Channel Activations)", use_container_width=True)
    with xai_col2:
        st.image(lime_colored, caption="LIME (Isolated Localized Boundaries)", use_container_width=True)
    with xai_col3:
        st.image(shap_colored, caption="SHAP (Global Importance Contribution Maps)", use_container_width=True)
