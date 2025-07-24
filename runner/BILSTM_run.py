import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATBiLSTM Model Runner")
    parser.add_argument("--information_type", required=False, type=str, help="Type of information (e.g., 'content', 'structure', 'both')")
    parser.add_argument("--content_size", required=False, type=int, help="Vocabulary size for content")
    parser.add_argument("--structure_size", required=False, type=int, help="Vocabulary size for structure")
    parser.add_argument("--max_length", required=False, type=int, help="Maximum length of sequences")
    parser.add_argument("--batch_size", required=False, type=int, help="Batch size for training")
    parser.add_argument("--epochs", required=False, type=int, help="Number of epochs for training")
    parser.add_argument("--embedding_dim", required=False, type=int, help="Number of embedding dimmension for training")
    parser.add_argument("--lstm_units", required=False, type=int, help="Number of lstm units for training")
    parser.add_argument("--attention_size", required=False, type=int, help="Number of attention size for training")
    parser.add_argument("--dropout_rate", required=False, type=float, help="Dropout rate for training")
    parser.add_argument("--train_data", type=str, default="balanced_dataset.csv", help="Path to the training data file")
    parser.add_argument("--test_data", type=str, default="csic_database_cleaned.csv", help="Path to the test data file")
    parser.add_argument("--vectorizer", type=str, default="tfidf", help="Vectorizer to use (e.g., 'tfidf', 'count')")
    parser.add_argument("--embedding_dropout_rate", type=float, default=0.5, help="Dropout rate for embedding layer")
    parser.add_argument("--LSTM_dropout_rate", type=float, default=0.5, help="Dropout rate for LSTM layer")
    parser.add_argument("--info_type", type=str, default="content", help="Type of information to use (e.g., 'content', 'structure', 'both')")

    args = parser.parse_args()
    data = args.train_data
    df = pd.read_csv(data)
    train_val, test = train_test_split(df, test_size=0.2, random_state=12345)
    kf = KFold(n_splits=5, shuffle=True, random_state=12345)
    if(args.info_type == "both"):
        from model.BiLSTM_TFIDF import BiLSTM_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            bilstm = BiLSTM_TFIDF(
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
            bilstm.build_model()
            bilstm.fit()
            bilstm.evaluate(fold)
            fold += 1
    elif(args.info_type == "structure"):
        from model.BiLSTM_Structure_TFIDF import BiLSTM_Structure_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            bilstm = BiLSTM_Structure_TFIDF(
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
            bilstm.build_model()
            bilstm.fit()
            bilstm.evaluate(fold)
            fold += 1
    else:
        from model.BiLSTM_Content_TFIDF import BiLSTM_Content_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            bilstm = BiLSTM_Content_TFIDF(
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
            bilstm.build_model()
            bilstm.fit()
            bilstm.evaluate(fold)
            fold += 1