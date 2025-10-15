# 🧠 Ensemble Models and Neural Networks: Customer Churn Prediction

**Author:** Kattya Contreras Valdés  
📍 Santiago, Chile  
🎓 *International Trade Engineer | Data Science*  
💻 *Power BI | Python | SQL | Statistical Models | Machine Learning | Neural Networks*  

---

## 📌 Project Overview

This project aims to **predict the probability of customer churn** in a telecommunications company, using **supervised Machine Learning techniques** within the module *Ensemble Models and Neural Networks*.

The objective is to **identify the key factors** influencing customer retention and attrition, building a **robust, balanced, and optimized predictive model** to support strategic decision-making in customer management.

---

## 🎯 Methodological Approach

The analysis follows these key stages:

### 1. Exploratory Data Analysis (EDA)
- Review of variables, missing values, and correlations  
- Visualization of behavioral patterns and customer segments

### 2. Preprocessing and Class Balancing (SMOTE)
- Address class imbalance between churned and retained customers  
- Normalize numerical variables

### 3. Predictive Modeling (Ensemble & Supervised ML)
- Implement **Decision Tree, Bagging, Random Forest**, and **MLP Neural Network**  
- Hyperparameter tuning with **GridSearchCV** to maximize performance

### 4. Model Evaluation
- Metrics: **F1-Score**, **ROC-AUC**, **OOB Score**  
- Feature importance analysis for churn prediction

### 5. Neural Network Baseline (MLP)
- Train a simple **Multi-Layer Perceptron** with *MLPClassifier* from scikit-learn  
- Serves as a baseline for ensemble methods and introduces a deep learning perspective

---

## 🧩 Technologies and Libraries

| Library | Purpose |
|---------|---------|
| **Python 3.11** | Programming language |
| **pandas, numpy** | Data manipulation & analysis |
| **matplotlib, seaborn** | Data visualization |
| **scikit-learn** | Modeling, metrics & optimization |
| **imbalanced-learn (SMOTE)** | Class balancing |
| **joblib** | Model persistence |

---

## 📈 Key Results

| Model | Main Parameters | F1-Score (Test) | ROC-AUC | OOB Score |
|-------|----------------|----------------|----------|------------|
| **Random Forest** | {'max_features': 'sqrt', 'n_estimators': 170} | **0.657** | **0.881** | **0.938** |
| **MLP Neural Network** | hidden_layer_sizes=(32,), activation='relu' | 0.642 | 0.864 | — |

> ✅ The **Random Forest** model achieved **high generalization and stability**, effectively identifying customers at higher churn risk and explaining the most relevant predictive features.

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
```


---

## 🔬 **Dataset Overview**

**Source:** `telecom_churn.csv`
**Shape:** (3333, 11)

**Target distribution:**

```
0 (retained): 85.5%
1 (churned): 14.5%
```

Most customers maintain their contracts (≈90%), and about 14.5% churn.
Key predictive factors include **Contract Renewal**, **Customer Service Calls**, and **Monthly Charges**.

---

## 📊 **Exploratory Insights**

* **High churn probability** is associated with customers who:

  * Do **not renew their contract**.
  * Have **frequent customer service calls**.
  * Have **higher monthly charges**.

* **Moderate churn correlation** observed with:

  * Day minutes and data usage.

---

## 🌲 **Model Performance Summary**

| Model                      | F1 Test  | ROC-AUC  |
| -------------------------- | -------- | -------- |
| Decision Tree (Baseline)   | 0.61     | 0.77     |
| Decision Tree + GridSearch | 0.68     | 0.81     |
| Bagging + SMOTE            | 0.68     | 0.87     |
| Heterogeneous Bagging      | 0.63     | 0.88     |
| Random Forest (45 trees)   | 0.67     | 0.88     |
| Random Forest (Optimized)  | **0.68** | **0.88** |

📊 **Conclusion:**
Regularization and ensemble techniques (Bagging, Random Forest) significantly improved **F1-Test** (≈0.67–0.68) and **ROC-AUC** (≈0.87–0.88), compared to individual models (F1≈0.61).

---

## 🔍 **Feature Importance (Optimized Random Forest)**

| Feature         | Importance |
| --------------- | ---------- |
| ContractRenewal | 0.20       |
| DayMins         | 0.16       |
| MonthlyCharge   | 0.16       |
| CustServCalls   | 0.12       |

> These variables are key indicators of customer dissatisfaction and churn risk.

---

## 📦 **Top 15 Customers with Highest Churn Probability**

The 15 customers with the **highest predicted churn probability** were exported to:

```bash
top15_churn_clients.csv
```

This dataset helps prioritize **customer retention actions** for at-risk users.

---

## 🧠 **Insights Summary**

* Ensemble methods outperform single models in both **stability and generalization**.
* The Random Forest model achieved an **OOB Score ≈ 0.94**, indicating high reliability.
* **SMOTE balancing** effectively corrected the target class imbalance (85/15 → 50/50).
* The MLP Neural Network provides a promising **deep learning baseline** for future work.

---

## 💾 **Future Enhancements**

* Implementation of **Stacking and XGBoost** models.
* Exploration of **Deep Neural Networks** using TensorFlow/Keras.
* Integration with **Power BI dashboards** for real-time churn monitoring.

---

## 📚 **License**

This project is licensed under the **MIT License**.

---

### 👩‍💻 **About the Author**

**Kattya Contreras Valdés**
Data Analyst | BI | Data Science
📍 Santiago, Chile
📧 [kattyacontreras.v@gmail.com](mailto:kattyacontreras.v@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/kattyacontrerasv)

> *"Transforming data into strategic insights that drive business decisions."*

```
