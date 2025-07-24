import tensorflow as tf
import pandas as pd
from tfidf.tfidf import tfidf_character_vocab_gen, text_to_index, padding_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import json
import os

class CNN_Content_TFIDF:
    def __init__(self, train, test, validation, max_length=300, vocab_size_content=10000, vocab_size_structure=10000, 
                 embedding_dim=128, filters=[16, 32, 64], kernel_sizes=[1, 2, 3], 
                 epochs=10, batch_size=32):
        self.max_length = max_length
        self.train = train
        self.test = test
        self.validation = validation
        self.vocab_size_content = vocab_size_content
        self.vocab_size_structure = vocab_size_structure
        self.embedding_dim = embedding_dim
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.content_data = None
        self.structure_data = None
        self.labels = None
        self.content_vocab = None
        self.structure_vocab = None

    def __build_model(self, max_length, vocab_size_content, embedding_dim=128):
        content_input = Input(shape=(max_length,), name='content_input')
        content_embed = Embedding(
            input_dim=vocab_size_content,
            output_dim=embedding_dim,
            name='content_embedding'
        )(content_input)
        
        content_convs = []
        for i, (filter_size, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            conv = Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                name=f'content_conv_{i}'
            )(content_embed)

            pool = MaxPooling1D(pool_size=max_length - kernel_size + 1, name=f'content_pool_{i}')(conv)
            content_convs.append(Flatten(name=f'content_flatten_{i}')(pool))
        
        if len(content_convs) > 1:
            content_features = concatenate(content_convs, name='content_features_concat')
        else:
            content_features = content_convs[0]
        
        dense = Dense(64, activation='relu', name='dense_1')(content_features)
        output = Dense(2, activation='softmax', name='classification_output')(dense)
        
        model = Model(
            inputs=[content_input], 
            outputs=[output]
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',   
            metrics='accuracy'
        )
        
        return model

    def build_model(self):
        train = self.train
        self.content_train = train['url']
        self.labels_train = train['label']

        test = self.test
        self.content_test = test['url']
        self.labels_test = test['label']

        validation = self.validation
        self.content_val = validation['url']
        self.labels_val = validation['label']

        dataset = pd.concat([train, test, validation], axis=0,ignore_index=True)
        self.content_data = dataset['url']
        self.labels = dataset['label']
        
        encoder = LabelEncoder()
        numerical_labels = encoder.fit_transform(self.labels)
        self.labels = to_categorical(numerical_labels, num_classes=2)
        self.labels = np.array(self.labels)

        self.labels_train = encoder.transform(self.labels_train)
        self.labels_train = to_categorical(self.labels_train, num_classes=2)
        self.labels_train = np.array(self.labels_train)
        self.labels_val = encoder.transform(self.labels_val)
        self.labels_val = to_categorical(self.labels_val, num_classes=2)
        self.labels_val = np.array(self.labels_val)
        self.labels_test = encoder.transform(self.labels_test)
        self.labels_test = to_categorical(self.labels_test, num_classes=2)
        self.labels_test = np.array(self.labels_test)

        df_train = pd.DataFrame({
            'url': self.content_train,
            'feature': self.content_train,
        })
        self.content_train, _, self.content_vocab, _, self.content_vectorizer, _ = tfidf_character_vocab_gen(df_train,max_len=self.max_length, max_features_content=self.vocab_size_content, max_features_structure=self.vocab_size_content)
        self.model = self.__build_model(
            self.max_length, 
            len(self.content_vocab), 
            self.embedding_dim
        )
        self.model.summary()

    def fit(self):
        self.content_val = text_to_index(self.content_val, self.content_vocab, self.vocab_size_content)
        self.content_val = padding_sequences(
            self.content_val, 
            max_len=self.max_length,
            padding='post',
            truncating='post'
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True, 
            verbose=1
        )
        
        history = self.model.fit(
            [self.content_train],
            self.labels_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(
                [self.content_val],
                self.labels_val
            ), 
            callbacks=[early_stopping, lr_scheduler], 
            shuffle=True
        )
        
            
        self.model.save('./h5/CNN_Content_TFIDF_model.h5')
        
        return history
    
    def evaluate(self, fold):
        self.content_test = text_to_index(self.content_test, self.content_vocab, self.vocab_size_content)
        self.content_test = padding_sequences(self.content_test, 
                                            max_len=self.max_length,
                                            padding='post',
                                            truncating='post'
                                            )
        y_pred_probs = self.model.predict([self.content_test])
        y_true = np.argmax(self.labels_test, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_probs[:, 1]),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        print("\nComprehensive Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC-ROC: {metrics['auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

        if not os.path.exists('./outputs/predictions'):
            os.makedirs('./outputs/predictions', exist_ok=True)
            
        filters_str = '-'.join(map(str, self.filters))
        kernels_str = '-'.join(map(str, self.kernel_sizes))
        
        with open(f'./outputs/predictions/evaluation_metrics_CNN_Content_TFIDF_epoch-{self.epochs}_embdim-{self.embedding_dim}_filters-{filters_str}_kernels-{kernels_str}_fold-{fold}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics