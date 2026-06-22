"""
eda.py
------
Exploratory Data Analysis for the SMS Spam Collection dataset.
Run this script after downloading the dataset to generate all EDA plots.

Generates:
  - plots/class_distribution.png
  - plots/message_length_distribution.png
  - plots/wordcloud_spam.png
  - plots/wordcloud_ham.png
  - plots/top_words_comparison.png
  - plots/character_count_boxplot.png
"""
 


import os, sys
 
# ── Bulletproof path setup ─────────────────────────────────────────────────────
# Get the absolute path of THIS file, then go one level up to project root
THIS_FILE = os.path.abspath(__file__)          # .../AI_ML_Project/notebooks/eda.py
NOTEBOOKS_DIR = os.path.dirname(THIS_FILE)     # .../AI_ML_Project/notebooks/
ROOT = os.path.dirname(NOTEBOOKS_DIR)          # .../AI_ML_Project/
SRC  = os.path.join(ROOT, 'src')              # .../AI_ML_Project/src/
 
# Insert src/ at the front of the path
if SRC not in sys.path:
    sys.path.insert(0, SRC)
 
# Uncomment the next line to debug if you still get errors:
# print("SRC path:", SRC, "| exists:", os.path.exists(SRC))
# print("preprocess_text.py exists:", os.path.exists(os.path.join(SRC, 'preprocess_text.py')))
 
import importlib.util, types
 
# Load preprocess_text.py directly from its file path — 100% reliable
_spec = importlib.util.spec_from_file_location(
    "preprocess_text",
    os.path.join(SRC, "preprocess_text.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_and_clean = _mod.load_and_clean
 
# ── Rest of imports ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
 
DATA_PATH = os.path.join(ROOT, 'data', 'SMSSpamCollection')
PLOTS_DIR = os.path.join(ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
 
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150
 
 
def run_eda():
    df = load_and_clean(DATA_PATH)
 
    df['char_count']  = df['message'].str.len()
    df['word_count']  = df['message'].str.split().str.len()
    df['has_url']     = df['message'].str.contains(r'http|www|\.com', case=False).astype(int)
    df['has_number']  = df['message'].str.contains(r'\d{4,}').astype(int)
    df['exclamation'] = df['message'].str.count('!')
 
    print(df[['label','char_count','word_count','has_url','has_number','exclamation']]
          .groupby('label').mean().round(2))
 
    # Plot 1: Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['label'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=['#27ae60','#e74c3c'], edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11)
    ax.set_title('Class Distribution: Ham vs Spam', fontsize=13)
    ax.set_ylabel('Number of messages')
    ax.set_ylim(0, max(counts.values) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'))
    plt.close()
    print("saved: class_distribution.png")
 
    # Plot 2: Message length
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label in zip(axes, ['char_count','word_count'], ['Character count','Word count']):
        for cls, color in [('ham','#27ae60'), ('spam','#e74c3c')]:
            ax.hist(df[df['label']==cls][col], bins=40, alpha=0.6,
                    label=cls.capitalize(), color=color, edgecolor='white')
        ax.set_title(f'{label} by class')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.legend()
    fig.suptitle('Message Length: Spam vs Ham', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'message_length_distribution.png'))
    plt.close()
    print("saved: message_length_distribution.png")
 
    # Plot 3: Top words
    def top_words(texts, n=20):
        return Counter(' '.join(texts).split()).most_common(n)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, words, title, color in [
        (axes[0], top_words(df[df['label']=='spam']['cleaned_text']), 'Top words in SPAM', '#e74c3c'),
        (axes[1], top_words(df[df['label']=='ham']['cleaned_text']),  'Top words in HAM',  '#27ae60'),
    ]:
        wl, ct = zip(*words)
        ax.barh(list(reversed(wl)), list(reversed(ct)), color=color, edgecolor='white')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Frequency (after stemming)')
    plt.suptitle('Most Common Words After Preprocessing', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_words_comparison.png'))
    plt.close()
    print("saved: top_words_comparison.png")
 
    # Plot 4: Feature boxplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes,
        ['char_count','exclamation','word_count'],
        ['Character count','Exclamation marks','Word count']):
        parts = ax.boxplot(
            [df[df['label']=='ham'][col], df[df['label']=='spam'][col]],
            labels=['Ham','Spam'], patch_artist=True
        )
        for patch, color in zip(parts['boxes'], ['#27ae60','#e74c3c']):
            patch.set_facecolor(color); patch.set_alpha(0.4)
        ax.set_title(title, fontsize=11)
    plt.suptitle('Feature Distributions: Ham vs Spam', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_boxplots.png'))
    plt.close()
    print("saved: feature_boxplots.png")
 
    print(f"\nEDA complete. Plots saved to: {PLOTS_DIR}")
    print(f"Total: {len(df)} | Spam: {(df['label']=='spam').sum()} | Ham: {(df['label']=='ham').sum()}")
 
 
if __name__ == '__main__':
    run_eda()