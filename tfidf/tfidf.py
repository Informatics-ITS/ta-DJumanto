import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from scipy.sparse.dia import dia_matrix

def read_dataset(df):
    urls = df['url'].astype(str).tolist()
    features = df['feature'].astype(str).tolist()
    return urls, features


def custom_token(text):
    return re.findall(r'[^\s]+', text)

def build_vocab(data, max_features=10000):
    vectorizer = TfidfVectorizer(
        tokenizer=custom_token,
        token_pattern=None,
        use_idf=True,
        max_features=max_features,
    )   
    vectorizer.fit_transform(data)
    vocab = vectorizer.vocabulary_
    vocab["<UNK>"] = max_features - 1
    return vocab, vectorizer

def text_to_index(sequences, vocab, max_features):
    unk_index = vocab.get("<UNK>", max_features - 1)
    
    def to_index(seq):
        return [
            vocab[word] if word in vocab and vocab[word] < max_features else unk_index
            for word in seq.lower().split()
        ]
    
    return [to_index(seq) for seq in sequences]

def padding_sequences(sequences, max_len, padding, truncating):
    return pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating, value=-100)


def tfidf_character_vocab_gen(df, max_len=300, max_features_content=10000, max_features_structure=10000):
    content, structure = read_dataset(df)
    content_vocab, content_vectorizer = build_vocab(content, max_features=max_features_content)
    structure_vocab, structure_vectorizer = build_vocab(structure, max_features=max_features_structure)
    
    content_indexed = text_to_index(content, content_vocab, max_features_content)
    structure_indexed = text_to_index(structure, structure_vocab, max_features_structure)
    
    X_content = padding_sequences(content_indexed, max_len=max_len, padding='post', truncating='post')
    X_structure = padding_sequences(structure_indexed, max_len=max_len, padding='post', truncating='post')
    return X_content, X_structure, content_vocab, structure_vocab, content_vectorizer, structure_vectorizer