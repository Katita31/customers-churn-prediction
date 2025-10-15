# 🧠 Ensemble Models and Neural Networks: Customer Churn Prediction

**Author:** Kattya Contreras Valdés  
📍 Santiago, Chile  
🎓 *International Trade Engineer | Data Science*  
💻 *Power BI | Python | SQL | Statistical Models | Machine Learning | Neural Networks*

---

## 📌 Project Overview

This project aims to **predict the probability of customer churn** in a telecommunications company using **supervised Machine Learning techniques**.  

The goal is to **identify key factors** influencing customer retention and attrition, and build a **robust, balanced, and optimized predictive model** to support strategic decision-making.

---

## 🎯 Methodological Approach

The analysis follows these main steps:

### 1. Exploratory Data Analysis (EDA)
- Review variables, missing values, and correlations.  
- Visualize behavioral patterns and customer segments.

### 2. Preprocessing and Class Balancing (SMOTE)
- Address class imbalance between churned and retained customers.  
- Normalize numerical variables.

### 3. Predictive Modeling (Ensemble & Supervised ML)
- Implement **Decision Tree, Bagging, Random Forest**, and **MLP Neural Network**.  
- Tune hyperparameters using **GridSearchCV** for optimal performance.

### 4. Model Evaluation
- Metrics: **F1-Score**, **ROC-AUC**, **OOB Score**.  
- Analyze feature importance for churn prediction.

### 5. Neural Network Baseline (MLP)
- Train a **Multi-Layer Perceptron** with `MLPClassifier` (scikit-learn).  
- Provides a deep learning baseline for ensemble methods.

---

## 🧩 Technologies and Libraries

| Library | Purpose |
|--------|---------|
| **Python 3.11** | Programming language |
| **pandas, numpy** | Data manipulation & analysis |
| **matplotlib, seaborn** | Data visualization |
| **scikit-learn** | Modeling, metrics & optimization |
| **imbalanced-learn (SMOTE)** | Class balancing |
| **joblib** | Model persistence |

---

## 📈 Key Results

| Model | Parameters | F1-Score (Test) | ROC-AUC | OOB Score |
|-------|-----------|----------------|---------|-----------|
| **Random Forest** | `{'max_features': 'sqrt', 'n_estimators': 170}` | **0.657** | **0.881** | **0.938** |
| **MLP Neural Network** | `hidden_layer_sizes=(32,), activation='relu'` | 0.642 | 0.864 | — |

> ✅ The **Random Forest** model shows **high generalization and stability**, effectively identifying high-risk churn customers.

---

## 🧠 Example — First Neural Network Layer

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Simple Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# Evaluate model
y_pred = mlp.predict(X_test)
print("F1-Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
