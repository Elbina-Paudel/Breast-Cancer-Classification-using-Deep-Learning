# 🧠 Breast Cancer Classification using PyTorch

This project implements a deep learning-based pipeline to classify breast tumors as benign or malignant using the Wisconsin Breast Cancer dataset. Built from scratch using PyTorch, the project demonstrates exploratory data analysis (EDA), custom model architecture, training-validation pipelines, and performance analysis.

---

## 🔍 Problem Statement

Accurate early detection of breast cancer is critical for effective treatment. This project aims to predict tumor malignancy based on diagnostic features extracted from digitized images of fine needle aspirates (FNAs) of breast masses.

---

## 🧰 Tools & Technologies

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Libraries:** scikit-learn, pandas, matplotlib, seaborn, numpy

---
## 🔧 Project Workflow

### Step 1: Exploratory Data Analysis (EDA)
- Checked for null values and dropped unnecessary columns.
- Visualized class distributions and feature correlations.
- Standardized features using `StandardScaler`.

### Step 2: Data Preprocessing
- Encoded diagnosis labels (M = 1, B = 0).
- Split dataset into training, validation, and test sets (70-15-15).
- Converted data to PyTorch tensors and created DataLoaders.

### Step 3: Model Architecture
- Built a custom neural network using `nn.Module` with:
  - Input layer (30 features)
  - Two hidden layers (ReLU activations)
  - Dropout regularization
  - Output layer (binary classification using Sigmoid)

### Step 4: Training & Validation
- Defined loss function: Binary Cross Entropy Loss
- Optimizer: Adam
- Trained the model across epochs and tracked:
  - Training & validation loss
  - Accuracy and F1-score

### Step 5: Evaluation & Error Analysis
- Evaluated performance on the test set using:
  - Confusion matrix
  - Classification report (Precision, Recall, F1)
  - ROC-AUC Score
- Visualized training curves and model predictions.

---

## 🔁 Hyperparameter Tuning

- Experimented with:
  - Learning rates (e.g., 0.001, 0.0005)
  - Dropout rates (0.2 to 0.5)
  - Batch sizes (16, 32)
- Found optimal values by monitoring validation performance and avoiding overfitting.

---

## 📌 Key Learnings

- Gained practical experience building a deep learning pipeline using PyTorch from scratch.
- Understood the impact of dropout, learning rate, and model depth.
- Learned to evaluate models with not just accuracy, but F1-score and ROC-AUC.



