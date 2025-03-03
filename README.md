# ❤️ **Heart Disease Prediction using K-Nearest Neighbors (KNN)**  

**Skills:** KNN Algorithm, Data Preprocessing, Feature Scaling, Model Evaluation  

---

## 🚀 **Project Overview**  
This project implements **K-Nearest Neighbors (KNN)** for predicting **heart disease** based on key medical attributes. It is a **basic machine learning project** designed to **understand the working of KNN classification**.  

This project covers:  
✅ **Understanding the KNN algorithm for medical diagnosis**  
✅ **Preprocessing heart disease dataset**  
✅ **Training & evaluating a KNN classifier using Scikit-Learn**  
✅ **Optimizing KNN performance through hyperparameter tuning**  

📌 **Reference:** [GitHub - Siddharth1698 KNN Heart Disease](https://github.com/Siddharth1698/Machine-Learning-Codes/blob/main/knn_heart_disease/knn_heart_disease.ipynb)  

---

## 🎯 **Key Objectives**  
✔ **Predict whether a patient has heart disease based on medical features**  
✔ **Understand the effect of different K values in KNN classification**  
✔ **Train, evaluate, and optimize a KNN model for medical prediction**  

---

## 📊 **Dataset Overview: UCI Heart Disease Dataset**  
The dataset contains **patient medical data** used for heart disease diagnosis.  

📌 **Feature Overview:**  
- **age** – Patient's age  
- **sex** – Gender (1 = Male, 0 = Female)  
- **cp** – Chest pain type (categorical)  
- **trestbps** – Resting blood pressure (mm Hg)  
- **chol** – Serum cholesterol level (mg/dl)  
- **fbs** – Fasting blood sugar (>120 mg/dl: 1 = True, 0 = False)  
- **restecg** – Resting ECG results  
- **thalach** – Maximum heart rate achieved  
- **exang** – Exercise-induced angina (1 = Yes, 0 = No)  
- **oldpeak** – ST depression (exercise vs rest)  
- **slope** – Slope of peak exercise ST segment  
- **ca** – Number of major blood vessels colored by fluoroscopy  
- **thal** – Defect type  
- **target** – **(0 = No Heart Disease, 1 = Heart Disease)**  

✅ **Example: Loading the Dataset**  
```python
import pandas as pd

df = pd.read_csv("heart.csv")
df.head()
```

✅ **Checking for Missing Values**  
```python
df.info()
df.isnull().sum()
```
💡 **Observation:** The dataset has **no missing values**.  

✅ **Summary Statistics**  
```python
df.describe()
```

---

## 📈 **Exploratory Data Analysis (EDA)**  
We visualize **correlations** between patient features and heart disease risk.  

✅ **Example: Correlation Heatmap**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
```
💡 **Findings:**  
- **Chest pain type (cp), thalach (max heart rate), and oldpeak show strong correlation with heart disease.**  
- **Fasting blood sugar (fbs) has low correlation.**  

✅ **Example: Distribution of Heart Disease Cases**  
```python
sns.countplot(x="target", data=df)
plt.title("Heart Disease vs No Heart Disease Cases")
```
💡 **Insight:** The dataset is **balanced**, meaning the model won't be biased toward one class.  

✅ **Example: Box Plot of Maximum Heart Rate (thalach) by Disease Status**  
```python
sns.boxplot(x="target", y="thalach", data=df)
plt.title("Heart Rate vs Heart Disease")
```
💡 **Observation:**  
- **Patients with heart disease tend to have lower maximum heart rates.**  

---

## 🏗 **Feature Engineering & Data Preprocessing**  
We **standardize features** to improve KNN model performance.  

✅ **Train-Test Split**  
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

✅ **Feature Scaling using StandardScaler**  
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
💡 **Why?** – KNN relies on **distance-based calculations**, so feature scaling is crucial.  

---

## 🤖 **Training KNN Model for Heart Disease Prediction**  
We implement **KNN classification** using `sklearn.neighbors.KNeighborsClassifier`.  

✅ **Training the Model**  
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)  # Default k=5
knn.fit(X_train, y_train)
```

✅ **Making Predictions**  
```python
y_pred = knn.predict(X_test)
```

---

## 📊 **Model Evaluation & Performance Metrics**  
We assess KNN’s performance using **accuracy, precision, recall, and F1-score**.  

✅ **Accuracy Score**  
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2f}")
```

✅ **Confusion Matrix**  
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for KNN Model")
plt.show()
```

✅ **Precision, Recall, and F1-Score**  
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```
💡 **Findings:**  
- **High accuracy** suggests good performance.  
- **F1-score ensures balanced prediction for both heart disease & non-disease cases.**  

---

## 🔍 **Optimizing KNN with Hyperparameter Tuning**  
To improve accuracy, we **experiment with different values of K**.  

✅ **Finding the Best K Value**  
```python
import numpy as np
from sklearn.model_selection import cross_val_score

k_range = range(1, 21)
accuracy_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    accuracy_scores.append(scores.mean())

# Plot results
plt.plot(k_range, accuracy_scores, marker="o")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Optimizing K in KNN")
plt.show()
```
💡 **Result:** The best K is chosen based on the **elbow point** of the graph.  

✅ **Training the Model with Optimized K**  
```python
best_k = np.argmax(accuracy_scores) + 1
knn_optimized = KNeighborsClassifier(n_neighbors=best_k)
knn_optimized.fit(X_train, y_train)
```

---

## 🔮 **Future Enhancements**  
🔹 **Compare KNN with Logistic Regression & Decision Trees**  
🔹 **Apply PCA for dimensionality reduction before training**  
🔹 **Use GridSearchCV for automated hyperparameter tuning**  

---

## 🎯 **Why This Project Stands Out for ML & AI Roles**  
✔ **Explains KNN with a step-by-step medical application**  
✔ **Applies Feature Engineering & Model Optimization**  
✔ **Uses real-world heart disease prediction dataset**  
✔ **Evaluates Model Performance with Precision & Recall**  

---

## 🛠 **How to Run This Project**  
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/heart-disease-prediction-knn.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook "Heart Disease Prediction using KNN.ipynb"
   ```
