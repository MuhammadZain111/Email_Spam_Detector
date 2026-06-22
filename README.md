# 📧 Spam Email / SMS Detector

A classical NLP machine learning project that classifies messages as **spam** or **ham** (legitimate)
using TF-IDF vectorisation and Logistic Regression — no deep learning required.

---

##  Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
```
UCI SMS Spam Collection:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

→ Download and place `SMSSpamCollection` file in the `data/` folder.
  (It's a tab-separated file with no header — no renaming needed.)
```

### 3. Run EDA
```bash
python notebooks/eda.py
```
Generates distribution plots and top word charts in `plots/`.

### 4. Train the model
```bash
python src/train.py
```
Trains Naive Bayes and Logistic Regression, prints comparison metrics,
saves best model as `model.pkl`, and generates evaluation plots.

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` — classify any message in real time.

---

## 📊 Model Performance (on held-out 20% test set)

| Metric     | Naive Bayes | Logistic Regression |
|------------|-------------|---------------------|
| Accuracy   | 97.8%       | **98.5%**           |
| Precision  | 95.1%       | **97.2%**           |
| Recall     | 92.3%       | **94.8%**           |


> **Why precision matters:** A false positive (real email marked as spam) costs the user
> an important missed message. We optimise for high precision while keeping recall strong.

---

##ML Concepts Covered

| Concept | Where Used |
|---------|-----------|
| Text preprocessing (tokenisation, stopwords, stemming) | `preprocess_text.py` |
| TF-IDF vectorisation with n-grams | `train.py` → `build_vectorizer()` |
| Multinomial Naive Bayes | `train.py` |
| Logistic Regression | `train.py` |
| Train/test split with stratification | `train.py` |
| 5-fold cross-validation | `train.py` |
| Confusion matrix | `train.py` → `_plot_confusion_matrices()` |
| ROC curve & AUC | `train.py` → `_plot_roc_curves()` |
| Feature importance | `train.py` → `_plot_top_features()` |
| sklearn Pipeline | Combines vectoriser + classifier into one object |
| Model persistence (joblib) | `train.py` → `save_model()` |

---


##  Why This Project  ?

- Demonstrates **end-to-end ML pipeline** (data → cleaning → training → deployment)
- Real-world NLP use case everyone understands immediately
- Shows knowledge of **evaluation metrics beyond accuracy**
- **Streamlit deployment** — live demo-able in any interview
- Feature importance analysis shows understanding of *why* the model works
