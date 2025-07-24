#!/bin/bash
embedding_dims=(32 64 128)
lstm_units_list=(32 64 128)
attention_sizes=(64)
epochs_list=(10)
batch_sizes=(200)
dropout_values=(0.3)
info_types=("both" "content" "structure")

content_size=100000
structure_size=100000
max_length=(70)
is_mining=0
eps=0.8
top_n_words=10
min_samples=5

for info_type in "${info_types[@]}"; do
  for embedding_dim in "${embedding_dims[@]}"; do
    for lstm_units in "${lstm_units_list[@]}"; do
      for attention_size in "${attention_sizes[@]}"; do
        for epochs in "${epochs_list[@]}"; do
          for batch_size in "${batch_sizes[@]}"; do
            for emb_dropout in "${dropout_values[@]}"; do
              for lstm_dropout in "${dropout_values[@]}"; do
                for attn_dropout in "${dropout_values[@]}"; do
                 for length in "${max_length[@]}"; do
                  exp_id="info-${info_type}_emb-${embedding_dim}_lstm-${lstm_units}_att-${attention_size}_ep-${epochs}_bs-${batch_size}_dropE-${emb_dropout}_dropL-${lstm_dropout}_dropA-${attn_dropoout}"

                  echo "================================================================================="
                  echo "Running $exp_id with length $length"
                  echo "================================================================================="

                  python3 ATBILSTM_run.py --information_type "$info_type" --content_size "$content_size" --structure_size "$structure_size" --max_length "$length" --batch_size "$batch_size" --epochs "$epochs" --embedding_dim "$embedding_dim" --lstm_units "$lstm_units" --is_mining "$is_mining" --eps "$eps" --top_n_words "$top_n_words" --min_samples "$min_samples" --dropout_rate "$lstm_dropout"  --train_data $1
                 done
                done
              done
            done
          done
        done
      done
    done
  done
done