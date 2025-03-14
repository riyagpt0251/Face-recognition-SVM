# Face Recognition with SVM and PCA

![Face Recognition](https://upload.wikimedia.org/wikipedia/commons/4/4c/Image-PCA.png)

## 🚀 Overview
This project applies **Support Vector Machines (SVM)** and **Principal Component Analysis (PCA)** to recognize faces from the **Labeled Faces in the Wild (LFW)** dataset. The dataset consists of celebrity faces, and the goal is to classify them correctly.

## 🛠 Installation & Setup
### Open in Google Colab 🚀
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Install Dependencies
```bash
pip install numpy scikit-learn
```

## 📂 Import Libraries
```python
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
```

## 📊 Load Dataset
```python
print("Loading the LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
```

**Dataset Features:**
- **X:** Feature matrix (pixel values of images)
- **y:** Target labels (person identity)
- **target_names:** Names of the people in the dataset

```python
X = lfw_people.data  # Feature matrix
y = lfw_people.target  # Target labels
target_names = lfw_people.target_names  # Names of the classes
```

## 📌 Split Dataset
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

## 🔄 Preprocess Data
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 🎭 Apply PCA (Dimensionality Reduction)
```python
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

```python
print(f"Feature shape after PCA: {X_train.shape}")
```
Output:
```
Feature shape after PCA: (901, 160)
```

## 🤖 Train SVM Classifier
```python
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
print("Training the SVM classifier...")
svm_classifier.fit(X_train, y_train)
```

## 📈 Model Evaluation
```python
y_pred = svm_classifier.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

### 📊 Classification Report:
| Person            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| Ariel Sharon    | 0.64      | 0.78   | 0.71     | 23      |
| Colin Powell    | 0.79      | 0.83   | 0.81     | 71      |
| Donald Rumsfeld | 0.71      | 0.68   | 0.69     | 37      |
| George W Bush   | 0.89      | 0.90   | 0.89     | 159     |
| Gerhard Schroeder | 0.66   | 0.76   | 0.70     | 33      |
| Hugo Chavez     | 0.92      | 0.52   | 0.67     | 21      |
| Tony Blair      | 0.84      | 0.74   | 0.79     | 43      |

🎯 **Final Accuracy: 80.88%**

## 📌 Conclusion
- We successfully trained an **SVM classifier** with **PCA** for feature extraction.
- The model achieved **80.88% accuracy** on the test dataset.
- PCA reduced the feature space significantly while maintaining high accuracy.

## 📜 License
This project is open-source and available under the **MIT License**.

---
📌 **Author:** Your Name 👨‍💻 | 🌟 Don't forget to ⭐ the repo if you like this project! 🚀

