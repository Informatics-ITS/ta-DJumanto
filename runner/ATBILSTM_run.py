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
    parser.add_argument("--is_mining", type=bool, default=False, help="yes for mining patterns")
    parser.add_argument("--only_mining", type=bool, default=False, help="Only do pattern mining for spesific resource")
    parser.add_argument("--mining_resource", type=str, default="./outputs/predictions/predictions.xlsx", help="Path to the mining resource file")
    parser.add_argument("--attention_weights_mining_resource", type=str, default="./outputs/attention_outputs/attention_weights.npy", help="Path to the attention weights mining resource file")
    parser.add_argument("--attention_hidden_mining_resource", type=str, default="./outputs/attention_outputs/attention_hidden_state.npy", help="Path to the attention hidden mining resource file")
    parser.add_argument("--top_n_words", type=int, default=10, help="Top N words for pattern mining")
    parser.add_argument("--eps", type=float, default=0.8, help="Epsilon for pattern mining")
    parser.add_argument("--min_samples", type=int, default=3, help="Minimum samples for pattern mining")
    parser.add_argument("--embedding_dropout_rate", type=float, default=0.2, help="Dropout rate for embedding layer")
    parser.add_argument("--lstm_dropout_rate", type=float, default=0.2, help="Dropout rate for LSTM layer")
    parser.add_argument("--attention_dropout_rate", type=float, default=0.2, help="Dropout rate for attention layer")

    args = parser.parse_args()

    data = args.train_data
    df = pd.read_csv(data)
    train_val, test = train_test_split(df, test_size=0.2, random_state=12345)
    kf = KFold(n_splits=5, shuffle=True, random_state=12345)
    if( args.information_type == "both"):
        from model.ATBiLSTM import ATBiLSTM

        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            atbilstm = ATBiLSTM(train=train_fold,
                                validation=val_fold,
                                test=test,
                                vocab_size_content=args.content_size,
                                vocab_size_structure=args.structure_size,
                                max_length=args.max_length,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                embedding_dim=args.embedding_dim,
                                lstm_units=args.lstm_units,
                                is_mining=args.is_mining,
                                eps=args.eps,
                                min_samples=args.min_samples,
                                top_n_words=args.top_n_words,
                                attention_dropout_rate=args.attention_dropout_rate
                                )
            atbilstm.build_model()
            atbilstm.fit(fold)
            atbilstm.evaluate(fold=fold)
            fold += 1

    elif( args.information_type == "content"):
        from model.ATBiLSTM_Content import ATBiLSTM_Content
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            atbilstm = ATBiLSTM_Content(train=train_fold,
                                validation=val_fold,
                                test=test,
                                vocab_size_content=args.content_size,
                                max_length=args.max_length,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                embedding_dim=args.embedding_dim,
                                lstm_units=args.lstm_units,
                                attention_dropout_rate=args.attention_dropout_rate,
                                )
            atbilstm.build_model()
            atbilstm.fit()
            atbilstm.evaluate(fold=fold)
            fold += 1
    elif( args.information_type == "structure"):
        from model.ATBiLSTM_Structure import ATBiLSTM_Structure
        fold = 1
        for train_index, val_index in kf.split(train_val):
            train_fold = train_val.iloc[train_index]
            val_fold = train_val.iloc[val_index]
            atbilstm = ATBiLSTM_Structure(train=train_fold,
                                test=test,
                                validation=val_fold,
                                vocab_size_structure=args.structure_size,
                                max_length=args.max_length,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                embedding_dim=args.embedding_dim,
                                lstm_units=args.lstm_units,
                                attention_dropout_rate=args.attention_dropout_rate
                                )
            atbilstm.build_model()
            atbilstm.fit()
            atbilstm.evaluate(fold=fold)
            fold += 1