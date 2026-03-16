#  Fake News Detection using NLP and Deep Learning

##  Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline to classify news articles as **Fake (0)** or **Real (1)** using both classical machine learning and deep learning techniques.

Models implemented:
- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes
- GRU-based Deep Learning Model

The project includes exploratory data analysis, preprocessing, feature engineering, model evaluation, ROC analysis, confusion matrix, learning curves, and error analysis.

---

##  Dataset

- Source: Kaggle – Fake and Real News Dataset
- Total Samples: 44,898 articles
- Classes:
  - Fake: 23,481
  - Real: 21,417

Stratified train-test split (80-20) was used to maintain class balance.

---

##  Text Preprocessing

The following preprocessing steps were applied:

- Lowercasing
- HTML tag removal
- Removal of non-alphabet characters
- Stopword removal
- Lemmatization
- Tokenization (fitted only on training data)
- Padding (max length = 300)

Data leakage was strictly avoided by fitting vectorizers and tokenizers only on training data.

---

##  Exploratory Data Analysis

- Class distribution analysis
- Missing value checks
- Text length distribution
- WordCloud comparison between Fake and Real news

### WordCloud Insights

- Fake news articles often contain personality-driven and emotionally charged language.
- Real news articles show institutional and location-based terminology such as “White House”, “government”, and “official”.

---

##  Models Implemented

### 1️. TF-IDF + Logistic Regression

- Max features: 10,000
- Accuracy: **98.59%**

### 2️. TF-IDF + Naive Bayes

- Accuracy: **93.74%**

### 3️. GRU Deep Learning Model

Architecture:
- Embedding Layer (10,000 vocab size, 64 dimensions)
- GRU Layer (32 units)
- Dropout (0.3)
- Dense Output (Sigmoid activation)

Training:
- Epochs: 5
- Batch size: 32
- Early stopping (patience=2)
- Validation split: 20%

Test Performance:
- Accuracy: **98.95%**
- AUC Score: **0.9986**

---

## Model Evaluation

### Confusion Matrix
- False Positives: 54
- False Negatives: 40
- Total Misclassifications: 94 / 8980

### ROC Curve
- AUC Score: 0.9986

### Learning Curves
- Stable training
- No significant overfitting observed

---

##  Model Comparison

| Model                | Accuracy |
|----------------------|----------|
| Naive Bayes          | 93.74%   |
| Logistic Regression  | 98.59%   |
| GRU                  | 98.95%   |

### Observation

Classical machine learning models performed competitively. The GRU model provided a marginal improvement by capturing sequential dependencies in text.

This highlights that for structured NLP classification problems, simpler models can perform nearly as well as deep learning while being computationally efficient.

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- NLTK
- Matplotlib
- Seaborn
- WordCloud

---

##  How to Run

Clone the repository:

```
git clone https://github.com/yourusername/fake-news-detection-nlp.git
cd fake-news-detection-nlp
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the notebook:

```
jupyter notebook
```

Or open directly in Google Colab.

---

##  Key Learnings

- Importance of preventing data leakage
- Comparison between classical ML and deep learning
- Interpreting ROC & AUC
- Conducting structured error analysis
- Sequential modeling using GRU

---

##  Future Improvements

- Implement Transformer-based models (BERT)
- Hyperparameter tuning
- Model deployment using Flask or FastAPI
- Model explainability using SHAP or LIME

