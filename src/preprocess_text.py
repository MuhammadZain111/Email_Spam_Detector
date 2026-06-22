"""
preprocess_text.py
------------------
Text cleaning pipeline for spam detection.
Handles tokenisation, stopword removal, and stemming.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data on first run
def download_nltk_data():
    for resource in ['punkt', 'stopwords', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

download_nltk_data()

stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single message.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation and digits
      4. Tokenise
      5. Remove stopwords
      6. Stem each token

    Returns a single cleaned string ready for TF-IDF.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # 4. Tokenise
    tokens = text.split()

    # 5. Remove stopwords + very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 6. Stem
    tokens = [stemmer.stem(t) for t in tokens]

    return ' '.join(tokens)


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the SMS Spam Collection dataset and apply cleaning.

    Dataset format (tab-separated, no header):
      column 0: label ('ham' or 'spam')
      column 1: message text

    Returns a DataFrame with columns: label, message, label_encoded, cleaned_text
    """
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['label', 'message'],
        encoding='latin-1'      # dataset uses latin-1 encoding
    )

    # Drop any duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Encode label: spam=1, ham=0
    df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

    # Apply text cleaning
    print("Cleaning text... (may take ~10 seconds for 5k rows)")
    df['cleaned_text'] = df['message'].apply(clean_text)

    print(f"Dataset loaded: {len(df)} rows | "
          f"Spam: {df['label_encoded'].sum()} | "
          f"Ham: {(df['label_encoded'] == 0).sum()}")

    return df


if __name__ == '__main__':
    # Quick test
    sample = "WINNER!! As a valued customer, claim your FREE prize now! Call 08712300150"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
