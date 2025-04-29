
# Flower Image Classification  
**Student Name:** Sudhanshu Kumar Mishra  
**Registration Number:** 20221266  

---

##  Project Overview

This project aims to classify flower images into **10 distinct categories** based on subtle visual differences like petal shape, size, and color.  
The classification model is built using a **custom CNN based on ResNet architecture** and trained from scratch on the Kaggle flower dataset.

---

##  Project Structure

```bash
project_sudhanshu_kumar_mishra/
│
├── checkpoints/
│   └── final_weights.pth        # Best model saved after training
│
├── data/
│   ├── img01.jpg                 # Sample data images for testing
│   ├── img02.jpg
│   └── ...
│
├── dataset.py                    # Data augmentation and DataLoader definitions
├── model.py                      # Custom ResNet model architecture
├── train.py                      # Training and evaluation loop
├── predict.py                    # Inference function for batch prediction
├── config.py                     # Hyperparameters and configuration variables
├── interface.py                  # Standardized imports for grading
└── README.md                     # Project description and instructions
```

---

##  How to Run

1. **Install Dependencies**

```bash
pip install torch torchvision tqdm
```

2. **Train the Model**

```bash
from interface import the_trainer, TheModel, the_dataloader

model = TheModel()
the_trainer(model, the_dataloader)
```

3. **Predict on New Images**

```bash
from interface import the_predictor

image_paths = ["data/img01.jpg", "data/img02.jpg", ...]  # List of image file paths
predictions = the_predictor(image_paths)
print(predictions)
```


##  Model Architecture

- **Base**: Custom Residual Network (ResNet-inspired)
- **Enhancements**:
  - Residual Connections to avoid vanishing gradients
  - Squeeze-and-Excitation (SE) blocks for channel-wise feature recalibration
  - Data augmentations to improve model generalization
  - Label smoothing in loss function to reduce overconfidence
  - Cosine annealing learning rate scheduler
- **Final classifier**: Fully Connected Layers with Dropout Regularization

---

## Training Strategy

- **Loss Function**: CrossEntropyLoss with label smoothing (`0.1`)
- **Optimizer**: AdamW (Adam optimizer with decoupled weight decay)
- **Scheduler**: CosineAnnealingLR
- **Early Stopping**: Patience of 10 epochs without improvement
- **Checkpointing**: Best model saved automatically

---

## Data Description

- Source: Kaggle Flower Classification Dataset  
- Input: Colored flower images (JPEG format)  
- Output: Classification label (1 out of 10 categories)

**Data Augmentations** (during training):
- Random resized crops
- Random horizontal & vertical flips
- Random affine transformations
- Random color jitter
- Random rotations

---

---

## Notes

- All code is modularized as per project submission guidelines.
- Hyperparameters and input sizes are stored in `config.py`.
- `interface.py` standardizes all imports for automatic grading.

---



#  Thank you! 

---
