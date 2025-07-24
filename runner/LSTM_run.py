import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Model Runner")
    parser.add_argument("--vectorizer", type=str, default="tfidf", help="Vectorizer to use (tfidf or count)")
    parser.add_argument("--content_size", required=False, type=int, help="Vocabulary size for content")
    parser.add_argument("--structure_size", required=False, type=int, help="Vocabulary size for structure")
    parser.add_argument("--max_length", required=False, type=int, help="Maximum length of sequences")
    parser.add_argument("--batch_size", required=False, type=int, help="Batch size for training")
    parser.add_argument("--epochs", required=False, type=int, help="Number of epochs for training")
    parser.add_argument("--embedding_dim", required=False, type=int, help="Number of embedding dimmension for training")
    parser.add_argument("--lstm_units", required=False, type=int, help="Number of lstm units for training")
    parser.add_argument("--lstm_dropout_rate", required=False, type=float, help="Dropout rate for LSTM layer")
    parser.add_argument("--embedding_dropout_rate", required=False, type=float, help="Dropout rate for embedding layer")
    parser.add_argument("--test_data", type=str, default="csic_database_cleaned.csv", help="Path to the test data file")
    parser.add_argument("--vector_size", type=int, default=100, help="Vector size for Doc2Vec")
    parser.add_argument("--predict_data_only", type=bool,default=False, help="Only predict on new data without training")
    parser.add_argument("--info_type", type=str, default="content", help="Type of information to use (content or structure)")
    parser.add_argument("--train_data", type=str, default="balanced_dataset.csv", help="Path to the training data file")
    args = parser.parse_args()

    data = args.train_data
    df = pd.read_csv(data)
    train_val, test = train_test_split(df, test_size=0.2, random_state=12345)
    kf = KFold(n_splits=5, shuffle=True, random_state=12345)

    if (args.info_type == "both"):
        from model.LSTM_TFIDF import LSTM_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            lstm_model = LSTM_TFIDF(train=train_fold,
                                    validation=val_fold,
                                    test=test,
                                    vocab_size_content=args.content_size,
                                    vocab_size_structure=args.structure_size,
                                    max_length=args.max_length,
                                    batch_size=args.batch_size,
                                    epochs=args.epochs,
                                    embedding_dim=args.embedding_dim,
                                    lstm_units=args.lstm_units
                                    )
            lstm_model.build_model()
            lstm_model.fit()
            lstm_model.evaluate(fold)
            fold += 1

    elif (args.info_type == "structure"):
        from model.LSTM_Structure_TFIDF import LSTM_Structure_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            lstm_model = LSTM_Structure_TFIDF(
                        train=train_fold,
                        validation=val_fold,
                        test=test,
                        vocab_size_content=args.content_size,
                        vocab_size_structure=args.structure_size,
                        max_length=args.max_length,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        embedding_dim=args.embedding_dim,
                        lstm_units=args.lstm_units
                        )
            lstm_model.build_model()
            lstm_model.fit()
            lstm_model.evaluate(fold)
            fold += 1
    else:
        from model.LSTM_Content_TFIDF import LSTM_Content_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            lstm_model = LSTM_Content_TFIDF(
                        train=train_fold,
                        validation=val_fold,
                        test=test,
                        vocab_size_content=args.content_size,
                        vocab_size_structure=args.structure_size,
                        max_length=args.max_length,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        embedding_dim=args.embedding_dim,
                        lstm_units=args.lstm_units
                        )
            lstm_model.build_model()
            lstm_model.fit()
            lstm_model.evaluate(fold)
            fold += 1