#!/bin/bash

embedding_dims=(32 64 128)
lstm_units_list=(32 64 128)
dropout_values=(0.0)
info_types=("structure" "content" "both")
epochs=10
batch_size=200
max_length=70
vectorizer="tfidf"
content_size=100000
structure_size=100000

for embedding_dim in "${embedding_dims[@]}"; do
  for lstm_units in "${lstm_units_list[@]}"; do
    for dropout in "${dropout_values[@]}"; do
      for type in "${info_types[@]}"; do

      echo "================================================================================="
      echo "Running emb-${embedding_dim}_lstm-${lstm_units}_dropE-${dropout}_dropL-${dropout}"
      echo "================================================================================="

      python3 LSTM_run.py \
        --vectorizer $vectorizer \
        --content_size $content_size \
        --structure_size $structure_size \
        --max_length $max_length \
        --batch_size $batch_size \
        --epochs $epochs \
        --embedding_dim $embedding_dim \
        --lstm_units $lstm_units \
        --embedding_dropout_rate $dropout \
        --lstm_dropout_rate $dropout \
        --info_type $type \
        --train_data balanced_dataset.csv

      echo ""
      done
    done
  done
done