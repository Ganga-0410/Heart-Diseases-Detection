# 💓 Heart Disease Detection using Machine Learning

This project aims to detect the presence of heart disease in patients using various machine learning classification algorithms. The dataset used is based on the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease), and the goal is to classify whether a patient has heart disease or not (binary classification).

---

## 📁 Dataset

The dataset contains **14 features** commonly associated with heart conditions such as:

- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting electrocardiographic results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise induced angina (exang)
- ST depression (oldpeak)
- Slope of peak exercise ST segment (slope)
- Number of major vessels colored by fluoroscopy (ca)
- Thalassemia (thal)
- Target (num) — 0 = No disease, 1 = Presence of disease

> **Note:** The `num` column has been converted into a binary target:  
> `0` → No heart disease  
> `1` → Heart disease

---

## 🔍 Project Workflow

1. **Data Cleaning & Preprocessing**
   - Null value imputation using `SimpleImputer`
   - Label encoding for categorical features
   - Target variable binarization

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots
   - Correlation analysis
   - Joint plots for feature relationships

3. **Model Building**
   - K-Nearest Neighbors (KNN) with Elbow Method
   - Logistic Regression
   - Decision Tree Classifier
   - (Optional: Add Random Forest, SVM, etc.)

4. **Model Evaluation**
   - Classification Report (precision, recall, F1-score)
   - Confusion Matrix heatmaps (combined in a single plot)
   - Cross-validation scores for performance consistency

---

## 📊 Visualizations Included

- Histograms and joint plots for EDA
- Heatmaps for null values and confusion matrices
- Elbow method for optimal `k` in KNN

---

## ✅ Results & Accuracy

Each model's performance is evaluated and compared. Cross-validation ensures the model is not overfitting and performs well on unseen data.

---

## 🛠️ Technologies Used

- Python 3
- pandas, NumPy, seaborn, matplotlib
- scikit-learn (classification models, preprocessing, metrics)
