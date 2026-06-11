# 🛡️ Multi-Class Deepfake Forensic Verification Suite

An advanced, production-grade Explainable AI (XAI) verification system built to detect and classify digital facial manipulations. This platform analyzes uploaded images to identify their underlying generative frameworks and maps out specific structural and pixel-level anomalies using cross-framework feature attributions.

🌐 **Live Dashboard Link:** [Access the Live Web Application](https://deepfake-ke5d2uxsctvcxnjjckq95v.streamlit.app/)

---

## 🚀 Key Features

- **Multi-Class Forensic Classification:** Categorizes facial inputs into four distinct classes:
  - `Real`: Authentic human portraits.
  - `StyleGAN`: Synthetic identities generated via GAN architectures.
  - `Diffusion`: Iterative denoising prompt-based image generation (e.g., Stable Diffusion, Midjourney).
  - `FaceSwap`: Identity replacement/manipulation mapped onto pre-existing imagery.
- **Ensemble Interpretability Panel (XAI Matrix):** Computes and cross-examines feature attributions using an ensemble of three independent explainability methods:
  - **Grad-CAM Thermal Focus:** Highlights the high-level convolutional neural network activation layers determining the classification outcome.
  - **LIME Feature Segments:** Extracts image superpixels to map localized regional contributions.
  - **GradientSHAP Pixel Densities:** Pinpoints precise pixel-level distribution shifts.
- **Unified Ensemble Blend:** Synthesizes all three attribution heatmaps into a single, comprehensive forensic map for streamlined diagnostic analysis.

---

## 🛠️ Architecture & Core Stack

- **Model Engine:** Deep Convolutional Network utilizing a custom-trained `EfficientNet-B0` backbone architecture.
- **Frontend Presentation:** Implemented with `Streamlit` for smooth, layout-driven interactive web dashboards.
- **Deep Learning Infrastructure:** Evaluated and engineered natively using `PyTorch` and `Torchvision`.
- **Explainability Integrations:** Built utilizing specialized diagnostic libraries: `Captum` (for Grad-CAM & SHAP) and `LIME`.

---

## 📋 Repository File Structure

```text
├── app.py                      # Main Streamlit web application script
├── requirements.txt            # Explicit python package dependencies
├── best_deepfake_model.pth     # Custom trained forensic network weights
└── README.md                   # Project documentation file
