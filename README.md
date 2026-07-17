# Attention-Based Hybrid CNN-Transformer Architecture for Multi-Label Chest X-Ray Diagnosis

## Overview

This project presents a hybrid deep learning framework for automated multi-label chest X-ray disease diagnosis. The proposed architecture combines **DenseNet-121** and **Vision Transformer (ViT-B/16)** using an **attention-based feature fusion mechanism** to leverage both local and global image representations.

The model is designed to improve diagnostic performance by dynamically balancing convolutional and transformer features for each disease class. It was developed as part of my **M.Tech in Artificial Intelligence** at **Indian Institute of Technology Jodhpur**.

---

## Features

- Hybrid CNN + Vision Transformer architecture
- Attention-based feature fusion
- Transfer Learning using pretrained models
- Selective Fine-Tuning of deeper layers
- Multi-label classification for 14 thoracic diseases
- Weighted Binary Cross-Entropy Loss
- Mixed Precision Training
- Disease-wise attention mechanism
- External validation on NIH ChestX-ray14 dataset

---

## Model Architecture

```
                 Chest X-ray Image
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   DenseNet-121                     Vision Transformer
   (Local Features)                 (Global Features)
        │                                 │
        └──────────────┬──────────────────┘
                       │
             Feature Projection
                       │
          Attention-Based Fusion
                       │
          Disease-wise Feature Learning
                       │
              Fully Connected Layer
                       │
        Multi-label Disease Prediction
```

---

## Dataset

### Training Dataset
- CheXpert Dataset
- Curated dataset of approximately 100,000 chest X-ray images

### External Validation
- NIH ChestX-ray14 Dataset

### Diseases Predicted

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Enlarged Cardiomediastinum
- Fracture
- Lung Lesion
- Lung Opacity
- No Finding
- Pleural Effusion
- Pleural Other
- Pneumonia
- Pneumothorax
- Support Devices

---

## Technologies Used

- Python
- PyTorch
- TorchXRayVision
- timm
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Training Strategy

- DenseNet-121 initialized with CheXpert pretrained weights
- Vision Transformer (ViT-B/16) initialized using pretrained weights
- Selective fine-tuning of deeper layers
- Differential learning rates
- AdamW Optimizer
- Weighted Binary Cross Entropy Loss
- Gradient Clipping
- Mixed Precision Training
- Early Stopping

---

## Results

The proposed hybrid model achieved improved diagnostic performance compared with standalone DenseNet-121 and Vision Transformer models.

Highlights include:

- Improved disease-wise recall
- Better diagnostic sensitivity
- Better balance of local and global feature learning
- Strong generalization on the NIH ChestX-ray14 dataset

---

## Project Structure

```
Hybrid-ChestXray/

├── app.py
├── hybrid_model_new/
├── notebooks/
├── images/
├── report/
├── requirements.txt
└── README.md
```

---

## Sample Outputs

Repository includes:

- Training Loss Curves
- Validation Loss Curves
- Disease-wise Recall Graphs
- Grad-CAM Visualizations
- Top-3 Disease Predictions
- Diagnostic Performance Comparison

---

## Future Improvements

- Real-time clinical deployment
- Explainable AI dashboard
- Multi-modal learning with clinical metadata
- Lightweight model for edge devices
- Integration with PACS systems

---

## Research Contribution

This work demonstrates that combining convolutional neural networks with transformer-based architectures using an adaptive attention fusion mechanism improves multi-label chest X-ray disease classification by effectively leveraging complementary local and global feature representations.

---

## Author

**Anish Subramanian Iyer**

M.Tech in Artificial Intelligence

Indian Institute of Technology Jodhpur

---

## License

This repository is intended for educational and research purposes.
