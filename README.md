# 🫁 Pediatric Pneumonia Diagnosis from Chest X-Ray Images

Binary classification (**NORMAL vs PNEUMONIA**) on pediatric chest X-ray images, comparing a classic Machine Learning model (SVM) with a Convolutional Neural Network based on Transfer Learning (Xception).

Project developed for the **Machine Learning and Deep Learning** exam, A.Y. 2025/2026 — [Master's Degree in Computer Engineering - Artificial Intelligence & Human-Computer Interaction](https://corsi.unige.it/), University of Genoa (Università degli Studi di Genova).

**Author:** Marta Nasso

---

## 📌 Context and Objective

Pneumonia is one of the leading causes of mortality in children under 5 years old. Early diagnosis through chest X-ray (CXR) is essential, but requires specialized personnel that is often unavailable in resource-limited areas.

The goal of the project is to build an automatic binary classification system, **NORMAL vs PNEUMONIA**, to support pediatric clinical diagnosis, with a particular focus on minimizing **false negatives** (missed pneumonia cases), the most clinically dangerous type of error.

## 🗂️ Dataset

- **Source:** [Kaggle — paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Clinical origin:** Guangzhou Women and Children's Medical Center
- **Total images:** ~5,856 pediatric chest X-rays
- **Classes:** NORMAL (~26%, 1,583 images) · PNEUMONIA (~74%, 4,273 images)
- **Format:** JPEG, grayscale (loaded as RGB), resized to 150×150 px

The original dataset had an insufficient Validation Set (only 16 images). All images were therefore merged into a single directory and re-split from scratch using a **70/15/15** ratio (Training/Validation/Test), preserving the class distribution via `stratify`.

## 🔬 Methodology

The project follows a two-stage path: first a **classic Machine Learning baseline**, then a move to **Deep Learning**, in order to contextualize and architecturally justify the improvement obtained.

### 1. ML Baseline — Support Vector Machine
- Images flattened from 150×150×3 into vectors of 67,500 features
- Linear kernel, trained on a subsample of 640 images (due to RAM limits and the SVM's quadratic/cubic complexity)
- **Test Set accuracy: 91.56%**, Pneumonia Recall: 0.94

### 2. Deep Learning — CNN with Transfer Learning (Xception)
- **Xception** backbone pre-trained on ImageNet (20.86M parameters)
- Data Augmentation (`RandomFlip` + `RandomRotation`)
- Pixel normalization to `[-1, +1]` (the range required by Xception)
- `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1)` (logit)
- Two-stage training:
  1. **Feature Extraction** (frozen backbone, 2,049 trainable parameters) → Val Acc 90.16%
  2. **Fine-Tuning** (unfrozen backbone, 20.81M trainable parameters, learning rate `1e-5` to avoid *catastrophic forgetting*) → **Val Acc 94.44%**

### 3. Evaluation & Explainable AI
- Confusion matrix, ROC curve (**AUC = 0.981**)
- **Grad-CAM** for model interpretability: visualizing the lung regions the CNN focuses on, enabling visual validation by medical staff

## 📊 Key Results

| Metric | SVM (Baseline) | CNN — Xception + Fine-Tuning |
|---|---|---|
| Accuracy | 91.56% | ~95%+ (Val 94.44%) |
| Pneumonia Recall | 0.94 (subsample) | Higher, on the full set |
| Training set used | 640 images | ~4,100 images (full set) |
| Spatial invariance | No (flat vector) | Yes (convolutional layers) |
| Interpretability | Support Vectors | Grad-CAM (visual) |
| Knowledge Transfer | No | ImageNet (20.86M parameters) |

The CNN with Xception and Fine-Tuning outperforms the SVM baseline on all relevant criteria. The SVM nonetheless confirms the feasibility of the task, architecturally justifying the move to Deep Learning to achieve clinically reliable performance.

## 📁 Repository Structure

```
Pneumonia-ML/
│
├── images/                              # Evaluation plots, metrics and XAI maps
│   ├── CM-DL.png                        # Deep Learning model confusion matrix
│   ├── CurvaROC.png                     # ROC curve and performance analysis
│   ├── GRADCAM.png                      # Grad-CAM activation maps
│   ├── Training.png                     # Training loss and accuracy curves
│   └── examples.png                     # Visual samples from the dataset
│
├── notebooks/                           # Executable Jupyter Notebooks
│   ├── ML&DL.ipynb                      # Main Deep Learning and Training pipeline
│   └── classification_Lecture.ipynb     # Supporting/experimentation notebook
│
├── presentation/                        # Official documentation and slides
│   ├── presentazione.pdf                # Project presentation slides (PDF)
│   └── presentazione_Marta_Nasso.html   # Slides in HTML format
│
└── README.md                            # Project documentation (this file)
```

## 🛠️ Technologies and Libraries

- **Python 3**
- **TensorFlow / Keras** — CNN construction and training, Transfer Learning with Xception
- **scikit-learn** — SVM, classification metrics (`classification_report`, `confusion_matrix`, `roc_curve`)
- **NumPy**
- **Matplotlib / Seaborn** — visualizations, training curves, Grad-CAM
- **Kaggle API** — dataset download

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Pneumonia-ML
   ```
2. Open the notebook `notebooks/ML&DL.ipynb` (using Google Colab is recommended for GPU access)
3. Upload your own `kaggle.json` credentials when prompted, for automatic dataset download
4. Run the cells in sequence: preprocessing → SVM baseline → CNN construction → training (feature extraction + fine-tuning) → evaluation → Grad-CAM

## 📄 Presentation

The full project presentation (clinical context, methodology, results and conclusions) is available in PDF and HTML format in the [`presentation/`](./presentation) folder.

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.
