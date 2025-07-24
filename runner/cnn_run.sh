#!/bin/bash

embedding_dims=(32 64 128)
embedding_dropouts=(0.0)
dense_dropouts=(0.0)
types=("both" "content" "structure")
for emb_dim in "${embedding_dims[@]}"; do
  for drop_emb in "${embedding_dropouts[@]}"; do
    for drop_dense in "${dense_dropouts[@]}"; do
        for type in "${types[@]}"; do
      echo "================================================================================="
      echo "Running CNN_emb-${emb_dim}_dropE-${drop_emb}_dropD-${drop_dense}"
      echo "================================================================================="

      python3 CNN_run.py \
        --max_length 70 \
        --vocab_size_content 100000 \
        --vocab_size_structure 100000 \
        --embedding_dim ${emb_dim} \
        --epochs 10 \
        --batch_size 200 \
        --embedding_dropout_rate ${drop_emb} \
        --dense_dropout_rate ${drop_dense} \
        --vectorizer tfidf \
        --info_type $type \
        --train_data balanced_dataset.csv
      echo ""
        done
    done
  done
done