#!/bin/bash

CONTENT_SIZE=100000
STRUCTURE_SIZE=100000
MAX_LENGTH=70
EPOCHS=10
EMBEDDING_DROPOUT=0.0
LSTM_DROPOUT=0.0

for INFO_TYPE in content both structure
do
  for BATCH_SIZE in 200
  do
    for EMBEDDING_DIM in 32 64 128
    do
      for LSTM_UNITS in 32 64 128
      do
        echo "=========================================="
        echo "Training config:"
        echo "Info Type         : $INFO_TYPE"
        echo "Batch Size        : $BATCH_SIZE"
        echo "Embedding Dim     : $EMBEDDING_DIM"
        echo "LSTM Units        : $LSTM_UNITS"
        echo "=========================================="

        python3 BILSTM_run.py \
          --vectorizer tfidf \
          --info_type "$INFO_TYPE" \
          --content_size "$CONTENT_SIZE" \
          --structure_size "$STRUCTURE_SIZE" \
          --max_length "$MAX_LENGTH" \
          --batch_size "$BATCH_SIZE" \
          --epochs "$EPOCHS" \
          --embedding_dim "$EMBEDDING_DIM" \
          --lstm_units "$LSTM_UNITS" \
          --embedding_dropout_rate "$EMBEDDING_DROPOUT" \
          --LSTM_dropout_rate "$LSTM_DROPOUT" \
          --train_data balanced_dataset.csv

        echo "✅ Done training: $INFO_TYPE | BS=$BATCH_SIZE | ED=$EMBEDDING_DIM | LSTM=$LSTM_UNITS"
        echo ""
      done
    done
  done
done