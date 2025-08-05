![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg) ![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)  

## Description  
This Jupyter notebook walks through the analysis and benchmarking of several Machine Learning algorithms for classifying breast tumors (malignant vs. benign) using the **Breast Cancer** dataset of wasiqaliyasir from Kaggle.  
It covers:  
- Downloading and loading the data via **kagglehub**  
- Data preprocessing (cleaning and encoding)  
- Exploratory Data Analysis (EDA)  
- Training and evaluating multiple models  
- Comparative visualization of performance (accuracy & AUC)  

## Table of Contents  
1. [Data Download & Loading](#data-download--loading)  
2. [Preprocessing](#preprocessing)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
4. [Model Benchmarking](#model-benchmarking)  
5. [Results Visualization](#results-visualization)  
7. [Requirements](#requirements)  
8. [Authors & License](#authors--license)  

---

## Data Download & Loading  
The first code block:  
- Imports `kagglehub`  
- Downloads the `wasiqaliyasir/breast-cancer-dataset` folder  
- Reads the CSV file `Breast_cancer_dataset.csv` into a pandas DataFrame  

```python
import kagglehub
path = kagglehub.dataset_download("wasiqaliyasir/breast-cancer-dataset")
import pandas as pd, os
csv_file = "Breast_cancer_dataset.csv"
df = pd.read_csv(os.path.join(path, csv_file))
display(df)
```  

## Preprocessing  
- Drop unnecessary columns (`id`, `Unnamed: 32`)  
- Encode the target (`M` → 1, `B` → 0)  

```python
# Clean up
df = df.drop(["id", "Unnamed: 32"], axis=1)
# Encode target
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```  

## Exploratory Data Analysis (EDA)  
Plot distributions of numerical features and compute skewness:  

```python
import seaborn as sns
import matplotlib.pyplot as plt

numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(X_train[feature], kde=True)
    plt.title(f"{feature} | Skewness: {X_train[feature].skew():.2f}")
plt.tight_layout()
plt.show()
```  

## Model Benchmarking  
We evaluate several algorithms using `GridSearchCV` optimizing for accuracy:

| Model                  | Key Hyperparameters                                            |
|------------------------|---------------------------------------------------------------|
| Logistic Regression    | `C`, `penalty`, `solver`                                      |
| SVM                    | `C`, `kernel`, `gamma`, `degree`                              |
| Decision Tree          | `max_depth`, `criterion`                                      |
| Random Forest          | `n_estimators`, `max_depth`, `criterion`                      |
| AdaBoost               | `n_estimators`, `learning_rate`                               |
| K-Nearest Neighbors    | `n_neighbors`, `weights`, `p`                                 |
| MLPClassifier          | `activation`, `solver`, `alpha`, `learning_rate_init`, etc.   |
| XGBoost                | `n_estimators`, `max_depth`, `learning_rate`, `subsample`     |

Each model is trained on `X_train`/`y_train`, tuned via grid search, and evaluated on `X_test`/`y_test`.  

## Results Visualization  
Compare performance (Accuracy vs. AUC):  

```python
models = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models]
aucs = [results[m]['auc'] for m in models]

plt.figure(figsize=(10, 6))
# ... plotting code for bar charts
```  

## Requirements  
```text
python>=3.8
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
kagglehub
```  

## Authors & License  
- **Author**: Charles Prioux  
- **License**: MIT License  