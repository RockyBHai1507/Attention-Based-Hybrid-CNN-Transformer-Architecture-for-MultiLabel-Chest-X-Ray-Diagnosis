import streamlit as st
import pandas as pd
import torch
import timm
from PIL import Image, ImageOps
import os
import numpy as np
import gc
import torchvision.transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. CONFIG & LABELS ---
st.set_page_config(page_title="M.Tech Medical AI Portal", layout="wide")

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

CLINICAL_KNOWLEDGE = {
    "Cardiomegaly": "The cardiothoracic ratio appears increased (>0.5), suggesting possible heart failure.",
    "Edema": "Increased vascular marking and cephalization noted, consistent with pulmonary congestion.",
    "Pleural Effusion": "Blunting of the costophrenic angles observed, indicating fluid accumulation.",
    "Atelectasis": "Linear opacities suggest partial lung collapse; clinical correlation advised.",
    "Pneumonia": "Focal consolidation or patchy opacities detected; check for clinical symptoms.",
    "No Finding": "The cardiomediastinal silhouette and both lungs appear within normal limits."
}

# --- 2. MODEL LOADING (C: DRIVE PATH) ---
@st.cache_resource
def load_trained_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=14)
    model_path = r"C:\Users\ansiy\Desktop\Chexpert_web_app\checkpoints\best_vit_model.pth"
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        st.error(f"❌ Model file not found at: {model_path}")
    model.eval()
    return model

# --- 3. PREPROCESSING & RESHAPE ---
medical_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

# --- 4. SIDEBAR ---
st.sidebar.title("📁 Patient Information")
patient_id = st.sidebar.text_input("Enter Patient ID (e.g., 64547)", placeholder="Enter number only")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

# --- 5. MAIN UI ---
st.title("🏥 Smart Chest X-Ray Diagnostic System")

if uploaded_file:
    try:
        model = load_trained_model()
        raw_img = Image.open(uploaded_file).convert('RGB')
        enhanced_img = ImageOps.autocontrast(raw_img) 
        tensor_input = medical_transform(enhanced_img).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor_input)
            probs = torch.sigmoid(output).numpy()[0]

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("📸 X-Ray Visualization")
            st.image(enhanced_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            
            if st.button("🔥 Generate AI Attention Heatmap"):
                with st.spinner("Analyzing transformer attention..."):
                    # Determine focus target
                    top_class = np.argmax(probs)
                    target_idx = np.argsort(probs)[-2] if LABELS[top_class] == "No Finding" else top_class
                    
                    # Grad-CAM for ViT Logic
                    target_layers = [model.blocks[-1].norm1]
                    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
                    
                    # Generate and process heatmap
                    grayscale_cam = cam(input_tensor=tensor_input, targets=[ClassifierOutputTarget(target_idx)])[0, :]
                    
                    # Fix: Use the enhanced original for the background, but convert to float 0-1
                    img_bg = np.array(enhanced_img.resize((224, 224))) / 255.0
                    
                    # Create overlay (The fix for the blue washout)
                    viz = show_cam_on_image(img_bg, grayscale_cam, use_rgb=True, image_weight=0.7)
                    
                    st.image(viz, caption=f"AI Focus: {LABELS[target_idx]} (Heatmap Overlay)", use_container_width=True)
                    del cam
                    gc.collect()

        with col2:
            st.subheader("📋 Verification & Ground Truth")
            valid_csv_path = r"D:\Mtech_Project_Data_2\CheXpert-v1.0-small\valid.csv"
            
            if os.path.exists(valid_csv_path) and patient_id:
                df_valid = pd.read_csv(valid_csv_path)
                
                # CLEANER SEARCH: Ensures it finds 'patient64555' even if you just type '64555'
                search_term = f"patient{patient_id}" if "patient" not in patient_id.lower() else patient_id
                match = df_valid[df_valid['Path'].str.contains(search_term, case=False, na=False)]
                
                if not match.empty:
                    actual_data = match.iloc[0]
                    st.success(f"✅ Record Found: Patient {patient_id}")
                    st.markdown("**Expert Radiologist Findings:**")
                    
                    # FIXED LOGIC: Removed the undefined 'actual_labels' variable
                    doctor_findings = [l for l in LABELS if actual_data.get(l) == 1.0]
                    
                    if doctor_findings:
                        for f in doctor_findings: 
                            st.error(f"● {f} (Confirmed)")
                    else:
                        st.info("● Result: Normal (No Pathologies Found)")
                        
                    # Optional: Show Uncertain findings if they exist
                    uncertain = [l for l in LABELS if actual_data.get(l) == -1.0]
                    if uncertain:
                        for u in uncertain:
                            st.warning(f"● {u} (Uncertain/Inconclusive)")
                else:
                    st.warning(f"🔍 No entry found in CSV for ID: {patient_id}")
            
            elif not patient_id:
                st.info("ℹ️ Enter a Patient ID in the sidebar to load database ground truth.")

            st.divider()
            
            st.subheader("🤖 AI Diagnostic Report")
            top_3_idx = np.argsort(probs)[-3:][::-1]
            detected_for_insights = []
            
            for i in top_3_idx:
                p = float(probs[i])
                st.write(f"**{LABELS[i]}**")
                st.progress(p)
                st.write(f"Confidence: {p*100:.2f}%")
                if p > 0.30:
                    detected_for_insights.append(LABELS[i])

            st.divider()
            
            st.subheader("📝 Clinical Insights")
            if detected_for_insights:
                with st.expander("View AI Observations", expanded=True):
                    for label in detected_for_insights:
                        desc = CLINICAL_KNOWLEDGE.get(label, "Clinical correlation required.")
                        st.markdown(f"● **{label}**: {desc}")

        gc.collect()

    except Exception as e:
        st.error(f"Runtime Error: {e}")
else:
    st.info("👈 Please upload an X-ray to begin.")