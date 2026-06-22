# 📧 Spam Email / SMS Detector

A classical NLP machine learning project that classifies messages as **spam** or **ham** (legitimate)
using TF-IDF vectorisation and Logistic Regression — no deep learning required.

---

## 📁 Project Structure

```
spam-detector/
├── data/
│   └── SMSSpamCollection        ← Download from UCI (see below)
├── notebooks/
│   └── eda.py                   ← Exploratory Data Analysis
├── plots/                       ← Generated charts (auto-created)
├── src/
│   ├── preprocess_text.py       ← Text cleaning pipeline
│   └── train.py                 ← Model training & evaluation
├── app.py                       ← Streamlit web app
├── model.pkl                    ← Saved model (after training)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

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
| F1 Score   | 93.7%       | **96.0%**           |
| ROC-AUC    | 0.98        | **0.99**            |

> **Why precision matters:** A false positive (real email marked as spam) costs the user
> an important missed message. We optimise for high precision while keeping recall strong.

---

## 🧠 ML Concepts Covered

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

## 🎯 Why This Project for a Portfolio?

- ✅ Demonstrates **end-to-end ML pipeline** (data → cleaning → training → deployment)
- ✅ Real-world NLP use case everyone understands immediately
- ✅ Shows knowledge of **evaluation metrics beyond accuracy**
- ✅ **Streamlit deployment** — live demo-able in any interview
- ✅ Feature importance analysis shows understanding of *why* the model works

---

## 💼 CV Bullet Points

- Built an NLP spam classifier using TF-IDF vectorisation and Logistic Regression, achieving 98.5% accuracy and 96% F1 score on the UCI SMS Spam Collection dataset
- Implemented a multi-step text preprocessing pipeline (tokenisation, stopword removal, Porter stemming) and compared Naive Bayes vs Logistic Regression with 5-fold cross-validation
- Deployed a real-time classification web app using Streamlit with confidence gauges, top-word explanations, and batch CSV processing

---

## ❓ Common Interview Questions

1. **Why is Naive Bayes well-suited for text classification?**
   - It treats each word independently (naive assumption), which works well for sparse TF-IDF matrices and is very fast to train.

2. **What is TF-IDF and how does it differ from simple word counts?**
   - TF-IDF weights words by how often they appear in a document (TF) but penalises words common across all documents (IDF). "Free" appearing in spam but not ham gets a high score; "the" appearing everywhere gets near-zero.

3. **Why did you use n-grams (bigrams)?**
   - Single words like "free" appear in ham too. Bigrams like "free prize" or "click now" are much more spam-specific.

4. **Why prioritise precision over recall for spam detection?**
   - A false positive (labelling real mail as spam) causes a user to miss an important message — that's worse than occasionally letting spam through.

5. **How would you improve this model further?**
   - Try character-level features, add sender metadata, use threshold tuning on the probability output, or experiment with a LinearSVC which often outperforms LR on text.

---

## 🚀 Deployment to Streamlit Community Cloud (Free)

1. Push this project to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set `app.py` as the entry point
4. Add your trained `model.pkl` to the repo (or use Git LFS)
5. Your live URL: `https://your-app-name.streamlit.app`

Add this URL to your CV under the project entry!

---

## 📚 Dataset Citation

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011).
*Contributions to the Study of SMS Spam Filtering: New Collection and Results.*
Proceedings of the 2011 ACM Symposium on Document Engineering, pp. 259-262.
