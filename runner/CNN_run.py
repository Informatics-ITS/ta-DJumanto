import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Model Runner")
    parser.add_argument("--dataset", type=str, default="balanced_dataset.csv", help="Path to the dataset")
    parser.add_argument("--max_length", type=int, default=300, help="Maximum length of sequences")
    parser.add_argument("--vocab_size_content", type=int, default=10000, help="Vocabulary size for content")
    parser.add_argument("--vocab_size_structure", type=int, default=10000, help="Vocabulary size for structure")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--filters", type=list, default=[16, 32, 64], help="Number of filters for CNN layers")
    parser.add_argument("--kernel_sizes", type=list, default=[3, 4, 5], help="Kernel sizes for CNN layers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--embedding_dropout_rate", type=float, default=0.5, help="Dropout rate for embedding layer")
    parser.add_argument("--dense_dropout_rate", type=float, default=0.5, help="Dropout rate for dense layer")
    parser.add_argument("--vectorizer", type=str, default="tfidf", help="Vectorizer to use (tfidf or count)")
    parser.add_argument("--test_data", type=str, default="csic_database_cleaned.csv", help="Path to the test data file")
    parser.add_argument("--info_type", type=str, default="content", help="Type of information to use (content, structure, or both)")
    parser.add_argument("--train_data", type=str, default="balanced_dataset.csv", help="Path to the training data file")

    args = parser.parse_args()

    data = args.train_data
    df = pd.read_csv(data)
    train_val, test = train_test_split(df, test_size=0.2, random_state=12345)
    kf = KFold(n_splits=5, shuffle=True, random_state=12345)
    if(args.info_type == "both"):
        from model.CNN_TFIDF import CNN_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            model = CNN_TFIDF(
                train=train_fold,
                validation=val_fold,
                test=test,
                max_length=args.max_length,
                vocab_size_content=args.vocab_size_content,
                vocab_size_structure=args.vocab_size_structure,
                embedding_dim=args.embedding_dim,
                filters=args.filters,
                kernel_sizes=args.kernel_sizes,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            model.build_model()
            model.fit()
            model.evaluate(fold)
            fold += 1
    elif(args.info_type == "structure"):
        from model.CNN_Structure_TFIDF import CNN_Structure_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            model = CNN_Structure_TFIDF(
                train=train_fold,
                validation=val_fold,
                test=test,
                max_length=args.max_length,
                vocab_size_content=args.vocab_size_content,
                vocab_size_structure=args.vocab_size_structure,
                embedding_dim=args.embedding_dim,
                filters=args.filters,
                kernel_sizes=args.kernel_sizes,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            model.build_model()
            model.fit()
            model.evaluate(fold)
            fold += 1
    else:
        from model.CNN_Content_TFIDF import CNN_Content_TFIDF
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            model = CNN_Content_TFIDF(
                train=train_fold,
                validation=val_fold,
                test=test,
                max_length=args.max_length,
                vocab_size_content=args.vocab_size_content,
                vocab_size_structure=args.vocab_size_structure,
                embedding_dim=args.embedding_dim,
                filters=args.filters,
                kernel_sizes=args.kernel_sizes,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            model.build_model()
            model.fit()
            model.evaluate(fold)
            fold += 1