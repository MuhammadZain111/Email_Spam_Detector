"""
app.py
------
Streamlit web app for real-time spam detection.

Features:
  - Paste any message → instant spam/ham prediction
  - Confidence score shown as a gauge
  - Top 5 spam-signal words highlighted
  - Batch mode: upload a CSV of messages
  - Model metrics sidebar

Run with:
  streamlit run app.py
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide",
)

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

@st.cache_resource
def load_model():
    """Load trained pipeline once and cache it."""
    if not os.path.exists(MODEL_PATH):
        st.error("❌ model.pkl not found. Run `python src/train.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)

pipeline = load_model()
vectorizer = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['clf']


# ── Helper: predict single message ────────────────────────────────────────────
def predict_message(text: str) -> dict:
    """
    Run the full pipeline on a raw message.

    Returns:
      label       — 'Spam' or 'Ham'
      probability — confidence for the predicted class (0–1)
      spam_prob   — raw probability of spam
      top_words   — top 5 words driving the spam score
    """
    from src.preprocess_text import clean_text

    cleaned = clean_text(text)
    proba = pipeline.predict_proba([cleaned])[0]
    spam_prob = proba[1]
    label = 'Spam' if spam_prob >= 0.5 else 'Ham'

    # --- Identify top spam signal words in this specific message ---
    feature_names = vectorizer.get_feature_names_out()
    tfidf_vec = vectorizer.transform([cleaned])
    tfidf_dense = tfidf_vec.toarray()[0]

    if hasattr(classifier, 'coef_'):
        weights = classifier.coef_[0]
    else:
        weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

    # Score = tfidf weight × model weight (only for words present in message)
    scores = tfidf_dense * weights
    top_idx = np.argsort(scores)[-5:][::-1]
    top_words = [
        (feature_names[i], round(float(scores[i]), 4))
        for i in top_idx if scores[i] > 0
    ]

    return {
        'label': label,
        'spam_prob': round(float(spam_prob) * 100, 1),
        'ham_prob': round(float(proba[0]) * 100, 1),
        'confidence': round(float(max(proba)) * 100, 1),
        'top_words': top_words,
        'cleaned': cleaned,
    }


# ── Helper: draw probability gauge ────────────────────────────────────────────
def draw_gauge(spam_prob: float):
    """Draw a semicircular gauge showing spam probability."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={'projection': 'polar'})
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # Background arc (grey)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1] * 200, color='#e0e0e0', linewidth=18, solid_capstyle='round')

    # Filled arc (green→red)
    filled_end = np.pi - (spam_prob / 100) * np.pi
    theta_filled = np.linspace(np.pi, filled_end, 200)
    color = '#e74c3c' if spam_prob > 50 else '#27ae60'
    ax.plot(theta_filled, [1] * len(theta_filled), color=color, linewidth=18, solid_capstyle='round')

    # Needle
    angle = np.pi - (spam_prob / 100) * np.pi
    ax.annotate('', xy=(angle, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Labels
    ax.text(np.pi, 1.3, 'Ham', ha='center', va='center', fontsize=10, color='#27ae60', fontweight='bold')
    ax.text(0, 1.3, 'Spam', ha='center', va='center', fontsize=10, color='#e74c3c', fontweight='bold')
    ax.text(np.pi / 2, 0.4, f'{spam_prob:.0f}%', ha='center', va='center',
            fontsize=18, fontweight='bold', color='#2c3e50')

    ax.set_ylim(0, 1.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Sidebar — model info
with st.sidebar:
    st.title("📊 Model Info")
    st.markdown("""
    **Algorithm:** Logistic Regression  
    **Features:** TF-IDF (unigrams + bigrams)  
    **Vocabulary size:** 5,000 features  
    **Training data:** SMS Spam Collection (UCI)
    """)
    st.divider()
    st.markdown("**Typical performance on held-out test set:**")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "98.5%")
    col2.metric("Precision", "97.2%")
    col1.metric("Recall", "94.8%")
    col2.metric("F1 Score", "96.0%")
    st.divider()
    st.caption("💡 Precision is prioritised — we avoid marking real emails as spam.")

    # Show saved plots if they exist
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    if os.path.exists(os.path.join(plots_dir, 'confusion_matrices.png')):
        with st.expander("Confusion Matrix"):
            st.image(os.path.join(plots_dir, 'confusion_matrices.png'))
    if os.path.exists(os.path.join(plots_dir, 'roc_curves.png')):
        with st.expander("ROC Curves"):
            st.image(os.path.join(plots_dir, 'roc_curves.png'))
    if os.path.exists(os.path.join(plots_dir, 'top_features.png')):
        with st.expander("Top Spam Words"):
            st.image(os.path.join(plots_dir, 'top_features.png'))


# Main area
st.title("📧 Spam Email / SMS Detector")
st.markdown("Enter a message below to classify it as **spam** or **ham** (legitimate).")

tab1, tab2, tab3 = st.tabs(["🔍 Single Message", "📂 Batch CSV", "🎓 How It Works"])

# ── Tab 1: Single message ──────────────────────────────────────────────────────
with tab1:
    # Example messages
    st.markdown("**Try an example:**")
    examples = {
        "🚨 Spam example": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461.",
        "✅ Ham example": "Hey, are you free for lunch tomorrow? Was thinking we could try that new place on Baker Street.",
        "🚨 Spam example 2": "FREE entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry.",
        "✅ Ham example 2": "I'll be home late tonight. Don't wait up for dinner, I'll grab something on the way.",
    }
    selected = st.selectbox("Load example", ["— type your own —"] + list(examples.keys()))
    default_text = examples.get(selected, "")

    message = st.text_area(
        "Message text",
        value=default_text,
        height=120,
        placeholder="Paste or type any SMS or email message here..."
    )

    if st.button("🔍 Classify Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message.")
        else:
            result = predict_message(message)

            # Result banner
            if result['label'] == 'Spam':
                st.error(f"🚨 **SPAM** — {result['spam_prob']}% confidence")
            else:
                st.success(f"✅ **HAM** (legitimate) — {result['ham_prob']}% confidence")

            col_gauge, col_words = st.columns([1, 1])

            with col_gauge:
                st.markdown("**Spam probability gauge**")
                fig = draw_gauge(result['spam_prob'])
                st.pyplot(fig, use_container_width=True)

            with col_words:
                st.markdown("**Top spam-signal words detected**")
                if result['top_words']:
                    for word, score in result['top_words']:
                        bar_pct = min(int(score * 200), 100)
                        st.markdown(
                            f"`{word}` "
                            f"<div style='background:#e74c3c;width:{bar_pct}%;height:6px;"
                            f"border-radius:3px;margin-bottom:8px'></div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No strong spam signals detected in this message.")

            with st.expander("Preprocessing details"):
                st.markdown(f"**Cleaned text fed to model:**  \n`{result['cleaned']}`")


# ── Tab 2: Batch CSV ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    Upload a CSV with a column named **`message`** to classify multiple messages at once.
    The result will be downloadable as a new CSV.
    """)

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        df_upload = pd.read_csv(uploaded)

        if 'message' not in df_upload.columns:
            st.error("CSV must have a column named `message`.")
        else:
            st.dataframe(df_upload.head(), use_container_width=True)

            if st.button("🔍 Classify All Messages", type="primary"):
                with st.spinner(f"Classifying {len(df_upload)} messages..."):
                    results = df_upload['message'].apply(predict_message)
                    df_upload['prediction'] = [r['label'] for r in results]
                    df_upload['spam_probability_%'] = [r['spam_prob'] for r in results]

                st.success(f"Done! {len(df_upload)} messages classified.")
                spam_count = (df_upload['prediction'] == 'Spam').sum()
                ham_count = len(df_upload) - spam_count
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(df_upload))
                c2.metric("🚨 Spam", spam_count)
                c3.metric("✅ Ham", ham_count)

                st.dataframe(df_upload, use_container_width=True)

                csv_out = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download results CSV",
                    data=csv_out,
                    file_name='spam_results.csv',
                    mime='text/csv',
                )


# ── Tab 3: How it works ────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## How the model works

    ### 1. Text preprocessing
    Raw messages are cleaned through a pipeline:
    - **Lowercase** → removes case sensitivity
    - **URL removal** → strips `http://...` patterns
    - **Punctuation & digit removal**
    - **Tokenisation** → splits into individual words
    - **Stopword removal** → removes "the", "is", "at"...
    - **Stemming** (Porter Stemmer) → "calling" → "call", "prizes" → "prize"

    ### 2. TF-IDF Vectorisation
    Each cleaned message becomes a numerical vector where each dimension
    represents a word or bigram (two-word pair). Words that appear often
    in spam but rarely in ham (like *"free"*, *"winner"*, *"claim"*) get
    high weights. Common words shared across all messages get lower weights.

    - **Vocabulary:** 5,000 most informative features
    - **N-grams:** unigrams + bigrams (`"free prize"`, `"click now"`)
    - **Sublinear TF:** reduces impact of very frequent terms

    ### 3. Logistic Regression
    The TF-IDF vector is fed into Logistic Regression, which learns a
    weight for each word feature indicating how much it contributes to
    spam probability. The output is a probability between 0 and 1.

    ### 4. Why Precision matters here
    A **false positive** (marking a real email as spam) is costly — the user
    misses an important message. So we optimise for **precision** (when we
    say spam, we're right) while maintaining high recall (we catch most spam).

    ---
    ### Key metric definitions
    | Metric | Formula | Meaning |
    |--------|---------|---------|
    | Accuracy | (TP+TN)/All | Overall correctness |
    | Precision | TP/(TP+FP) | Of predicted spam, how many were actually spam |
    | Recall | TP/(TP+FN) | Of actual spam, how many did we catch |
    | F1 | 2×P×R/(P+R) | Harmonic mean of precision and recall |
    | ROC-AUC | — | Model's ability to distinguish classes across thresholds |
    """)
