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

def _partial_fit(self, X):
        X = X.copy()
        tokenizer = self.tokenizer if hasattr(self, 'tokenizer') and self.tokenizer else None
        token_pattern = self.token_pattern if self.token_pattern else None
        for doc in X:
            if self.lowercase:
                doc = doc.lower()
            if tokenizer:
                tokens = tokenizer(doc)
            elif token_pattern:
                tokens = re.findall(token_pattern, doc)
            else:
                raise ValueError("No tokenizer or token_pattern is defined.")
            indices_to_insert = []
            for w in tokens:
                if w not in self.vocabulary_:
                    self.vocabulary_[w] = -1
                    tmp_keys = sorted(list(self.vocabulary_.keys()))
                    tmp_dict = {tmp_keys[i]: i for i in range(len(tmp_keys))}
                    self.vocabulary_ = {k: tmp_dict[k] for k in self.vocabulary_}

                    self._tfidf.n_features_in_ += 1
                    indices_to_insert.append(self.vocabulary_[w])

            doc_frequency = (self.n_docs + self.smooth_idf) / np.exp(
                self.idf_ - 1
            ) - self.smooth_idf

            for index_to_insert in indices_to_insert:
                doc_frequency = np.insert(doc_frequency, index_to_insert, 0)
            self.n_docs += 1

            for w in set(tokens):
                doc_frequency[self.vocabulary_[w]] += 1

            idf = (
                np.log(
                    (self.n_docs + self.smooth_idf) / (doc_frequency + self.smooth_idf)
                )
                + 1
            )
            self._tfidf.idf_ = idf
            self._tfidf._idf_diag = dia_matrix((idf, 0), shape=(len(idf), len(idf)))

def partial_fit_vectorizer(vectorizer, new_data):
    vectorizer.n_docs += len(new_data)
    vectorizer.partial_fit(X=new_data)
    vocab = vectorizer.vocabulary_
    vocab = {term: idx + 1 for term, idx in vocab.items()}
    vocab["<UNK>"] = max(vocab.values()) + 1
    return vocab

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
    
    max_content = max(idx for seq in content_indexed for idx in seq)
    assert max_content <= max_features_content, f"Content indices exceed vocab size! Found {max_content}"
    
    max_structure = max(idx for seq in structure_indexed for idx in seq)
    assert max_structure <= max_features_structure, f"Structure indices exceed vocab size! Found {max_structure}"
    
    X_content = padding_sequences(content_indexed, max_len=max_len, padding='post', truncating='post')
    X_structure = padding_sequences(structure_indexed, max_len=max_len, padding='post', truncating='post')
    return X_content, X_structure, content_vocab, structure_vocab, content_vectorizer, structure_vectorizer